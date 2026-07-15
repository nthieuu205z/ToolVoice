"""Chứng minh bản sinh theo LÔ tính ra đúng cùng con số với bản gốc từng-câu-một.

Đây là bài test đắt nhất trong bộ (nạp model thật ~15 giây, cần GPU) nhưng bắt buộc: sinh
theo lô đụng vào đệm trái, attention mask và position_ids — sai một li thì giọng đọc méo
đi theo kiểu tai người nghe ra mà không test nào bắt được. Nên ta không so audio, ta so
TRẠNG THÁI ẨN và LOGITS: chúng là tất định, và nếu chúng khớp thì phần lấy mẫu phía sau
(vốn dùng lại nguyên hàm _sample_token của thư viện) không thể lệch được.

Tự bỏ qua khi máy không có GPU/CUDA hoặc chưa tải model — đường ONNX/CPU không dùng file này.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Sinh theo lô chỉ chạy trên engine PyTorch/GPU"
)

CAU = [
    "Xin chào các bạn.",
    "Hôm nay chúng ta sẽ tìm hiểu về cách hoạt động của động cơ điện xoay chiều.",
    "Cảm ơn đã theo dõi và hẹn gặp lại.",
]


@pytest.fixture(scope="module")
def engine():
    from pipeline.model_store import is_ready, vieneu_spec

    if not is_ready(vieneu_spec()):
        pytest.skip("Model VieNeu chưa tải về máy")

    from vieneu import Vieneu

    tts = Vieneu()
    if tts.backend != "pytorch":
        pytest.skip("Không phải engine PyTorch")
    return tts


@pytest.fixture(scope="module")
def prompts(engine):
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    eng = engine.engine
    spk, ref = engine._resolve_ref(engine._default_voice, None, True, True)
    style_id = eng._resolve_style_id("tu_nhien")
    ps = [eng._build_prompt_2d(phonemize_text_with_emotions(c), None, ref, style_id)
          for c in CAU]
    # Câu phải KHÁC độ dài, nếu không thì bài test này không chạm tới phần đệm.
    assert len({p.shape[0] for p in ps}) > 1
    return ps, spk


def _prefill_goc(eng, prompt, spk):
    """Đúng cách engine gốc prefill một câu: không mask, không position_ids."""
    model = eng.model
    spk_t = eng._resolve_speaker_emb(spk)
    inp = prompt.unsqueeze(0).to(eng.device)
    embeds = model._build_inputs_embeds(inp, speaker_emb=spk_t)
    out = model.semantic_backbone(inputs_embeds=embeds, use_cache=True, return_dict=True)
    return out.last_hidden_state[:, -1]          # (1, H)


def _prefill_lo(eng, prompts, spk):
    """Prefill theo lô — chính đoạn nằm trong generate_codes_batch."""
    model, cfg, dev = eng.model, eng.config, eng.device
    B, n_vq = len(prompts), cfg.n_vq
    lens = [p.shape[0] for p in prompts]
    S = max(lens)

    padded = torch.full((B, S, n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long)
    padded[:, :, 0] = 0
    mask = torch.zeros((B, S), dtype=torch.long)
    for i, p in enumerate(prompts):
        padded[i, S - lens[i]:] = p
        mask[i, S - lens[i]:] = 1
    padded, mask = padded.to(dev), mask.to(dev)
    pos = (mask.cumsum(dim=1) - 1).clamp(min=0)

    spk_t = eng._resolve_speaker_emb(spk)
    if spk_t is not None:
        spk_t = spk_t.expand(B, -1).contiguous()

    embeds = model._build_inputs_embeds(padded, speaker_emb=spk_t)
    out = model.semantic_backbone(inputs_embeds=embeds, attention_mask=mask,
                                  position_ids=pos, use_cache=True, return_dict=True)
    return out.last_hidden_state[:, -1]          # (B, H)


@torch.no_grad()
def test_prefill_theo_lo_ra_dung_trang_thai_nhu_tung_cau_mot(engine, prompts):
    """Phép thử cốt lõi: đệm trái + mask + position_ids có tái hiện đúng bản gốc không.

    Sai ở đây nghĩa là câu ngắn bị RoPE đánh số vị trí lệch, hoặc token thật nhìn thấy ô
    đệm — cả hai đều làm giọng méo mà vẫn phát ra tiếng, nên không thể phát hiện bằng mắt.
    """
    ps, spk = prompts
    eng = engine.engine

    goc = torch.cat([_prefill_goc(eng, p, spk) for p in ps])      # (B, H)
    lo = _prefill_lo(eng, ps, spk)                                # (B, H)

    assert goc.shape == lo.shape
    # bfloat16 có ~3 chữ số thập phân; sai số cộng dồn qua các tầng nên nới nhẹ ngưỡng.
    lech = (goc.float() - lo.float()).abs().max().item()
    thang = goc.float().abs().max().item()
    assert lech / thang < 0.02, f"trạng thái ẩn lệch {lech:.4f} (thang {thang:.2f})"


@torch.no_grad()
def test_logits_eos_va_codebook_dau_khop_nhau(engine, prompts):
    """Từ cùng một trạng thái ẩn, bản lô phải cho cùng logits EOS và logits codebook 0.

    Logits là tất định (chưa qua lấy mẫu), nên đây là chỗ duy nhất so được chính xác.
    """
    from pipeline.vieneu_batch import _decode_one_frame_batched

    ps, spk = prompts
    eng = engine.engine
    model, cfg = eng.model, eng.config

    h = _prefill_lo(eng, ps, spk)                                 # (B, H)

    # Bản gốc: chạy từng hàng một qua decode_one_frame của thư viện.
    goc_txt = []
    for i in range(h.shape[0]):
        _, local = model.decode_one_frame(
            h[i : i + 1],
            text_token_id=torch.tensor([cfg.speech_generation_start_token_id], device=eng.device),
            temperature=0.8, top_k=25, audio_top_p=0.95, repetition_penalty=1.0,
            history_by_channel=None,
        )
        goc_txt.append(model.text_lm_head(local[0, 0]).float())

    # Bản lô: cả ba hàng một lượt.
    seen = [None] * cfg.n_vq
    _, local_lo = _decode_one_frame_batched(
        model, h, cfg.speech_generation_start_token_id, seen,
        temperature=0.8, top_k=25, top_p=0.95, repetition_penalty=1.0,
    )
    lo_txt = model.text_lm_head(local_lo[:, 0]).float()           # (B, V)

    for i, g in enumerate(goc_txt):
        lech = (g - lo_txt[i]).abs().max().item()
        thang = g.abs().max().item()
        assert lech / thang < 0.02, f"logits EOS hàng {i} lệch {lech:.4f}"
        # Quan trọng hơn cả sai số: cùng một token thắng, vì EOS quyết định bằng argmax.
        assert int(g.argmax()) == int(lo_txt[i].argmax())


@torch.no_grad()
def test_moi_cau_ra_dung_do_dai_cua_rieng_no(engine, prompts):
    """Mỗi hàng phải dừng ở EOS của chính nó, không bị cắt theo hàng ngắn nhất."""
    from pipeline.vieneu_batch import generate_codes_batch

    ps, spk = prompts
    codes = generate_codes_batch(engine.engine, ps, spk, max_new_frames=120)

    assert len(codes) == len(ps)
    for c in codes:
        assert c.shape[0] > 0, "có câu không sinh ra frame nào"
        assert c.shape[1] == engine.engine.config.n_vq
    # Câu dài nhất phải sinh nhiều frame hơn câu ngắn nhất — nếu bằng nhau là dấu hiệu
    # mọi hàng bị dừng chung một chỗ (lỗi kinh điển khi gộp lô).
    assert codes[1].shape[0] > codes[0].shape[0]
