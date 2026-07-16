"""Sinh giọng theo LÔ cho engine PyTorch của VieNeu — nơi hiệu năng thật sự nằm.

Vì sao cần: engine gốc sinh từng câu một, batch = 1. Vòng sinh token của nó bị trói bởi
CHI PHÍ PHÓNG KERNEL chứ không phải sức tính. Đo trên RTX 3090:

    một bước backbone ở batch  1 : 10,54 ms
    một bước backbone ở batch 64 : 11,65 ms   ← gần như không đắt thêm

Tức GPU tiêu 10 ms chỉ để nhận lệnh, rồi tính xong trong tích tắc. Nhồi 16 câu vào cùng
một kernel cho ~14,5 lần thông lượng mà không tốn thêm gì. Chạy nhiều tiến trình KHÔNG
thay được việc này: GPU GeForce không chạy song song nhiều CUDA context, driver chỉ chia
lượt thời gian — đo thật, 4 tiến trình và 12 tiến trình cho thông lượng y hệt nhau
(4,5x so với 4,3x), điện chỉ 160W trên card 350W.

Cách làm: dùng lại đúng các khối của thư viện (`_build_prompt_2d`, `_build_inputs_embeds`,
`acoustic_decoder.cached_step`, `_sample_token`, `_decode_codes`) — chúng vốn đã nhận batch
sẵn. Chỉ có `decode_one_frame` là hardcode `[0]`, nên ta viết lại đúng một hàm đó theo lô.
Phép toán không đổi một dấu; tests/test_vieneu_batch.py chứng minh bằng cách so trạng thái
ẩn của bản lô với bản gốc.

Đệm bên TRÁI (left-pad): các câu dài ngắn khác nhau, mà sau prefill ta cần lấy `[:, -1]`
làm trạng thái cuối. Đệm trái thì cột cuối luôn là token thật của MỌI hàng. Kèm theo phải
truyền `attention_mask` (để token thật không nhìn thấy phần đệm) và `position_ids` (RoPE
phải đếm từ token thật đầu tiên, không phải từ ô đệm).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def _sample_batched(model, ch: int, hidden: torch.Tensor, seen: list, *,
                    temperature: float, top_k: float, top_p: float,
                    repetition_penalty: float) -> torch.Tensor:
    """Lấy mẫu code cho codebook `ch` trên cả lô. Trả (B,)."""
    from vieneu._v3_turbo_engine.modeling_v3_turbo import _sample_token

    logits = model.audio_lm_heads[ch](hidden).float()   # (B, V)

    # Phạt lặp: bản gốc giữ một `set` Python cho mỗi codebook và gọi .item() từng code —
    # mỗi lần .item() là một lần ép GPU dừng chờ CPU (16 lần mỗi frame). Ở đây giữ nguyên
    # CÔNG THỨC đó nhưng bằng mặt nạ bool trên GPU, nên không còn lần chặn nào.
    if repetition_penalty != 1.0:
        if seen[ch] is None:
            seen[ch] = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
        phat = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        logits = torch.where(seen[ch], phat, logits)

    # Phạt đã áp ở trên rồi -> gọi hàm lấy mẫu của thư viện với penalty=1.0 để dùng CHÍNH
    # code của họ cho temperature/top_k/top_p (nó vốn chạy được trên tensor 2 chiều).
    code = _sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                         repetition_penalty=1.0)                      # (B,)

    if repetition_penalty != 1.0:
        seen[ch].scatter_(1, code.unsqueeze(1), True)
    return code


def _decode_one_frame_batched(model, h: torch.Tensor, sgs_id: int, seen: list, *,
                              temperature: float, top_k: float, top_p: float,
                              repetition_penalty: float):
    """Bản theo lô của model.decode_one_frame. Trả (frame_codes (B, n_vq), prefill_out (B, 2, H)).

    Bám từng dòng bản gốc (modeling_v3_turbo.py:202-236), chỉ bỏ các chỉ số [0] cứng.
    """
    dec = model.acoustic_decoder
    n_vq = model.config.n_vq
    H = model.config.hidden_size
    L = len(dec.layers)
    B = h.shape[0]
    dev = h.device
    local_dtype = next(dec.parameters()).dtype

    cond = h.to(dtype=local_dtype)                                        # (B, H)
    sgs = torch.full((B,), sgs_id, device=dev, dtype=torch.long)
    sgs = sgs.clamp(0, model.config.text_vocab_size - 1)
    txt = model.text_embeddings(sgs).to(dtype=local_dtype)                # (B, H)

    tok = torch.stack([cond, txt], dim=1)                                 # (B, 2, H)
    pos = torch.tensor([0, 1], device=dev)
    hidden, pk, pv = dec.cached_step(tok, pos, [None] * L, [None] * L)    # hidden (B, 2, H)
    prefill_out = hidden

    codes = [_sample_batched(model, 0, hidden[:, 1], seen, temperature=temperature,
                             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)]
    for ch in range(1, n_vq):
        emb = model.audio_embeddings[ch - 1](codes[-1]).to(dtype=local_dtype)   # (B, H)
        pos = torch.tensor([ch + 1], device=dev)
        hidden, pk, pv = dec.cached_step(emb.view(B, 1, H), pos, pk, pv)         # (B, 1, H)
        codes.append(_sample_batched(model, ch, hidden[:, 0], seen, temperature=temperature,
                                     top_k=top_k, top_p=top_p,
                                     repetition_penalty=repetition_penalty))

    return torch.stack(codes, dim=1), prefill_out                         # (B, n_vq)


@torch.no_grad()
def generate_codes_batch(engine, prompts: list[torch.Tensor], speaker_emb, *,
                         temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
                         max_new_frames: int = 300, repetition_penalty: float = 1.2,
                         ) -> list[torch.Tensor]:
    """Sinh code âm thanh cho NHIỀU prompt cùng lúc. Trả về list các (T_i, n_vq)."""
    model = engine.model
    cfg = engine.config
    dev = engine.device
    B = len(prompts)
    n_vq = cfg.n_vq
    audio_pad = cfg.audio_pad_token_id
    eos_id = cfg.speech_generation_end_token_id
    sgs_id = cfg.speech_generation_start_token_id

    # ── đệm TRÁI ────────────────────────────────────────────────────────
    lens = [p.shape[0] for p in prompts]
    S = max(lens)
    padded = torch.full((B, S, n_vq + 1), audio_pad, dtype=torch.long)
    padded[:, :, 0] = 0                       # kênh văn bản: id 0, an toàn với mọi vocab
    mask = torch.zeros((B, S), dtype=torch.long)
    for i, p in enumerate(prompts):
        padded[i, S - lens[i]:] = p           # dồn về bên PHẢI -> cột cuối là token thật
        mask[i, S - lens[i]:] = 1
    padded = padded.to(dev)
    mask = mask.to(dev)
    # RoPE phải đếm từ token thật đầu tiên: ô đệm không được chiếm mất vị trí 0.
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)

    spk = engine._resolve_speaker_emb(speaker_emb)          # (1, D) hoặc None
    if spk is not None:
        spk = spk.expand(B, -1).contiguous()

    # ── prefill ─────────────────────────────────────────────────────────
    embeds = model._build_inputs_embeds(padded, speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=embeds, attention_mask=mask,
                                  position_ids=pos, use_cache=True, return_dict=True)
    past = out.past_key_values
    h = out.last_hidden_state[:, -1]                        # đệm trái -> hàng nào cũng đúng
    cur_pos = pos[:, -1]                                    # (B,)

    # ── vòng sinh ───────────────────────────────────────────────────────
    seen: list = [None] * n_vq
    frames: list[torch.Tensor] = []
    lengths = torch.full((B,), max_new_frames, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    for t in range(max_new_frames):
        frame, local_out = _decode_one_frame_batched(
            model, h, sgs_id, seen, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty)
        frames.append(frame)                                # bản gốc GHI frame rồi mới xét EOS

        text_logits = model.text_lm_head(local_out[:, 0]).float()          # (B, V)
        eos = text_logits.argmax(dim=-1) == eos_id
        moi = eos & ~done
        lengths = torch.where(moi, torch.tensor(t + 1, device=dev), lengths)
        done = done | eos
        if bool(done.all()):                               # 1 lần đồng bộ/frame (gốc: ~18)
            break

        slot = torch.full((B, 1, n_vq + 1), audio_pad, dtype=torch.long, device=dev)
        slot[:, 0, 0] = sgs_id
        slot[:, 0, 1:] = frame
        slot_embed = model._build_inputs_embeds(slot, speaker_emb=spk)

        mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=dev)], dim=1)
        cur_pos = cur_pos + 1
        step = model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past,
                                       attention_mask=mask, position_ids=cur_pos.unsqueeze(1),
                                       use_cache=True, return_dict=True)
        past = step.past_key_values
        h = step.last_hidden_state[:, 0]

    if not frames:
        return [torch.zeros(0, n_vq, dtype=torch.long) for _ in range(B)]

    het = torch.stack(frames)                              # (T, B, n_vq)
    dai = lengths.clamp(max=het.shape[0]).tolist()
    return [het[: dai[i], i].cpu() for i in range(B)]


def synthesize_batch(v3turbo, texts: list[str], voice: str, style: str, *,
                     batch_size: int = 16, max_chars: int = 256,
                     apply_watermark: bool = False, **gen) -> list[np.ndarray]:
    """Bản theo lô của V3TurboVieNeuTTS.infer cho NHIỀU câu. Trả list sóng âm float32.

    Giữ nguyên đường đi của bản gốc (v3turbo.py:243-265): cắt câu thành chunk, phiên âm,
    sinh, rồi ghép lại kèm khoảng lặng theo loại ranh giới. Khác đúng một chỗ: mọi chunk
    của MỌI câu được gom lại rồi sinh theo lô, thay vì từng chunk một.
    """
    from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks
    from vieneu_utils.phonemize_text import (
        normalize_to_chunks_v3_with_gaps,
        phonemize_text_with_emotions,
    )

    engine = v3turbo.engine
    speaker_emb, ref_codes = v3turbo._resolve_ref(voice, None, True, True)
    style_id = engine._resolve_style_id(style)

    # Trải phẳng: một câu có thể thành nhiều chunk. Nhớ đường về để ghép lại.
    phang: list[torch.Tensor] = []
    thuoc_ve: list[int] = []
    gaps_theo_cau: list[list] = []
    for i, text in enumerate(texts):
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        gaps_theo_cau.append(gaps)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            phang.append(engine._build_prompt_2d(ph, None, ref_codes, style_id))
            thuoc_ve.append(i)

    if not phang:
        return [np.array([], dtype=np.float32) for _ in texts]

    # Sinh theo từng lô rồi giải mã TỪNG CÂU. Không gộp lô phần giải mã: codec dựng tensor
    # audio (B, C, T) đệm về câu dài nhất — với câu dài/nhiều câu là hàng chục GB, tràn VRAM
    # (đo thật: gộp lô đòi 27 GB trên card 12 GB → tràn). Giải mã từng câu chỉ ~9s/280 câu,
    # không phải nút cổ chai (nút nằm ở vòng sinh, xem _lo_theo_do_dai bên pipeline/tts.py).
    wavs: list[np.ndarray] = []
    for i in range(0, len(phang), batch_size):
        lo = phang[i : i + batch_size]
        codes = generate_codes_batch(engine, lo, speaker_emb, **gen)
        wavs.extend(engine._decode_codes(c) for c in codes)

    # Ghép chunk về đúng câu của nó.
    ket: list[np.ndarray] = []
    for i in range(len(texts)):
        cua_toi = [w for w, chu in zip(wavs, thuoc_ve) if chu == i]
        if not cua_toi:
            ket.append(np.array([], dtype=np.float32))
            continue
        wav = join_audio_chunks(cua_toi, v3turbo.sample_rate,
                                silence_ps=gaps_to_silence(gaps_theo_cau[i]))
        ket.append(v3turbo._apply_watermark(wav) if apply_watermark else wav)
    return ket
