"""Open and stream managed files without reopening a validated pathname."""

from __future__ import annotations

import errno
import hashlib
import mimetypes
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import formatdate
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Sequence
from urllib.parse import quote

import anyio
from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse, Response
from starlette.types import Receive, Scope, Send


@dataclass(frozen=True)
class OpenedOwnedFile:
    file: BinaryIO
    path: Path
    stat_result: os.stat_result


class _MalformedRangeHeader(Exception):
    def __init__(self, content: str = "Malformed range header.") -> None:
        self.content = content


class _RangeNotSatisfiable(Exception):
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size


class HeldFileResponse(Response):
    """Stream one validated open file and close it on every ASGI exit path."""

    chunk_size = 64 * 1024
    max_ranges = 100

    def __init__(
        self,
        opened: OpenedOwnedFile,
        *,
        filename: str,
        media_type: str | None = None,
    ) -> None:
        self.file = opened.file
        self.path = opened.path
        self.filename = filename
        self.stat_result = opened.stat_result
        try:
            resolved_media_type = (
                media_type
                or mimetypes.guess_type(filename or str(opened.path))[0]
                or "application/octet-stream"
            )
            encoded_filename = quote(filename)
            if encoded_filename != filename:
                disposition = f"attachment; filename*=utf-8''{encoded_filename}"
            else:
                disposition = f'attachment; filename="{filename}"'
            etag_base = f"{opened.stat_result.st_mtime}-{opened.stat_result.st_size}"
            headers = {
                "accept-ranges": "bytes",
                "content-disposition": disposition,
                "content-length": str(opened.stat_result.st_size),
                "last-modified": formatdate(opened.stat_result.st_mtime, usegmt=True),
                "etag": f'"{hashlib.md5(etag_base.encode(), usedforsecurity=False).hexdigest()}"',
            }
            super().__init__(
                content=b"",
                media_type=resolved_media_type,
                headers=headers,
            )
        except BaseException:
            self.close()
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            request_headers = Headers(scope=scope)
            http_range = request_headers.get("range")
            http_if_range = request_headers.get("if-range")
            send_header_only = scope["method"].upper() == "HEAD"

            if http_range is None or (
                http_if_range is not None and not self._should_use_range(http_if_range)
            ):
                await self._send_simple(send, send_header_only)
                return

            try:
                ranges = self._parse_range_header(http_range, self.stat_result.st_size)
            except _MalformedRangeHeader as exc:
                await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
                return
            except _RangeNotSatisfiable as exc:
                await PlainTextResponse(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{exc.max_size}"},
                )(scope, receive, send)
                return

            if not ranges:
                await self._send_simple(send, send_header_only)
            elif len(ranges) == 1:
                start, end = ranges[0]
                await self._send_single_range(
                    send, start, end, self.stat_result.st_size, send_header_only
                )
            else:
                await self._send_multiple_ranges(
                    send, ranges, self.stat_result.st_size, send_header_only
                )
        finally:
            # The response-level finally also covers normal completion and disconnects.
            self.close()

    async def _send_simple(self, send: Send, send_header_only: bool) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        await anyio.to_thread.run_sync(self.file.seek, 0)
        more_body = True
        while more_body:
            chunk = await anyio.to_thread.run_sync(self.file.read, self.chunk_size)
            more_body = len(chunk) == self.chunk_size
            await send({"type": "http.response.body", "body": chunk, "more_body": more_body})

    async def _send_single_range(
        self, send: Send, start: int, end: int, file_size: int, send_header_only: bool
    ) -> None:
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        await anyio.to_thread.run_sync(self.file.seek, start)
        more_body = True
        while more_body:
            chunk = await anyio.to_thread.run_sync(
                self.file.read, min(self.chunk_size, end - start)
            )
            start += len(chunk)
            more_body = len(chunk) == self.chunk_size and start < end
            await send({"type": "http.response.body", "body": chunk, "more_body": more_body})

    async def _send_multiple_ranges(
        self,
        send: Send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        boundary = secrets.token_hex(13)
        content_length, header_generator = self._generate_multipart(
            ranges, boundary, file_size, self.headers["content-type"]
        )
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(content_length)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        for start, end in ranges:
            await send(
                {
                    "type": "http.response.body",
                    "body": header_generator(start, end),
                    "more_body": True,
                }
            )
            await anyio.to_thread.run_sync(self.file.seek, start)
            while start < end:
                chunk = await anyio.to_thread.run_sync(
                    self.file.read, min(self.chunk_size, end - start)
                )
                if not chunk:
                    break
                start += len(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"\r\n", "more_body": True})
        await send(
            {
                "type": "http.response.body",
                "body": f"--{boundary}--".encode("latin-1"),
                "more_body": False,
            }
        )

    def _should_use_range(self, http_if_range: str) -> bool:
        return http_if_range in {self.headers["last-modified"], self.headers["etag"]}

    @classmethod
    def _parse_range_header(cls, http_range: str, file_size: int) -> list[tuple[int, int]]:
        try:
            units, range_value = http_range.split("=", 1)
        except ValueError:
            raise _MalformedRangeHeader() from None

        if units.strip().lower() != "bytes":
            raise _MalformedRangeHeader("Only support bytes range")
        if range_value.count(",") + 1 > cls.max_ranges:
            return []

        ranges = cls._parse_ranges(range_value, file_size)
        if not ranges:
            raise _MalformedRangeHeader("Range header: range must be requested")
        if any(not (0 <= start < file_size) for start, _ in ranges):
            raise _RangeNotSatisfiable(file_size)
        if any(start >= end for start, end in ranges):
            raise _MalformedRangeHeader("Range header: start must be less than end")
        if len(ranges) == 1:
            return ranges

        ranges.sort()
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            previous_start, previous_end = merged[-1]
            if start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _parse_ranges(range_value: str, file_size: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for part in range_value.split(","):
            part = part.strip()
            if not part or part == "-" or "-" not in part:
                continue
            start_value, end_value = (value.strip() for value in part.split("-", 1))
            try:
                start = int(start_value) if start_value else max(file_size - int(end_value), 0)
                end = (
                    int(end_value) + 1
                    if start_value and end_value and int(end_value) < file_size
                    else file_size
                )
            except ValueError:
                continue
            ranges.append((start, end))
        return ranges

    @staticmethod
    def _generate_multipart(
        ranges: Sequence[tuple[int, int]],
        boundary: str,
        max_size: int,
        content_type: str,
    ) -> tuple[int, Callable[[int, int], bytes]]:
        boundary_len = len(boundary)
        static_header_len = 49 + boundary_len + len(content_type) + len(str(max_size))
        content_length = sum(
            len(str(start))
            + len(str(end - 1))
            + static_header_len
            + end
            - start
            for start, end in ranges
        ) + 4 + boundary_len

        def header(start: int, end: int) -> bytes:
            return (
                f"--{boundary}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Range: bytes {start}-{end - 1}/{max_size}\r\n"
                "\r\n"
            ).encode("latin-1")

        return content_length, header

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


def open_owned_file(root: Path, value: str | Path, not_found_detail: str) -> OpenedOwnedFile:
    """Atomically open a direct regular child of a managed directory."""
    try:
        root_path, candidate_path = _direct_child_paths(root, value)
        if os.name == "nt":
            return _open_windows(root_path, candidate_path)
        return _open_posix(root_path, candidate_path)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise HTTPException(404, not_found_detail) from None


def owned_file_response(
    root: Path,
    value: str | Path,
    *,
    filename: str,
    not_found_detail: str,
    media_type: str | None = None,
) -> HeldFileResponse:
    opened = open_owned_file(root, value, not_found_detail)
    return HeldFileResponse(opened, filename=filename, media_type=media_type)


@contextmanager
def atomic_owned_file(root: Path, value: str | Path) -> Iterator[BinaryIO]:
    """Exclusively create a final direct child and hold it while writing."""
    root_path, candidate_path = _direct_child_paths(root, value)
    if os.name == "nt":
        with _owned_file_writer_windows(root_path, candidate_path) as file:
            yield file
    else:
        with _owned_file_writer_posix(root_path, candidate_path) as file:
            yield file


def _direct_child_paths(root: Path, value: str | Path) -> tuple[Path, Path]:
    root_path = Path(os.path.abspath(os.fspath(root)))
    candidate_path = Path(os.path.abspath(os.fspath(value)))
    if candidate_path.parent != root_path or candidate_path.name in {"", ".", ".."}:
        raise FileNotFoundError(errno.ENOENT, "Path is not a direct child", candidate_path)
    if os.name == "nt" and ":" in candidate_path.name:
        raise FileNotFoundError(errno.ENOENT, "Alternate data streams are not allowed", candidate_path)
    return root_path, candidate_path


def _open_posix(root: Path, candidate: Path) -> OpenedOwnedFile:
    """Use openat + O_NOFOLLOW so validation applies to the returned descriptor."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        # Fail closed on an unknown POSIX port; silently reopening by path would restore TOCTOU.
        raise OSError(errno.ENOTSUP, "Secure no-follow opens are unavailable")

    common_flags = getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | common_flags)
    file_fd = -1
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")

        file_flags = os.O_RDONLY | os.O_NOFOLLOW | common_flags | getattr(os, "O_NONBLOCK", 0)
        file_fd = os.open(candidate.name, file_flags, dir_fd=root_fd)
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError(errno.EPERM, "Managed result is not a regular file")

        entry_stat = os.stat(candidate.name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry_stat.st_mode) or (
            entry_stat.st_dev,
            entry_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise OSError(errno.EAGAIN, "Managed result changed while opening")

        file = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = -1
        return OpenedOwnedFile(file=file, path=candidate, stat_result=opened_stat)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


@contextmanager
def _owned_file_writer_posix(root: Path, candidate: Path) -> Iterator[BinaryIO]:
    """Exclusively create the final entry and leave failed partial writes in place."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError(errno.ENOTSUP, "Secure no-follow writes are unavailable")

    common_flags = getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | common_flags)
    file_fd = -1
    file: BinaryIO | None = None
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")

        file_fd = os.open(
            candidate.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | common_flags,
            0o600,
            dir_fd=root_fd,
        )

        opened_stat = os.fstat(file_fd)
        owned_identity = (opened_stat.st_dev, opened_stat.st_ino)
        entry_stat = os.stat(candidate.name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            raise OSError(errno.EPERM, "Managed target is not a regular file")
        if opened_stat.st_nlink != 1 or entry_stat.st_nlink != 1:
            raise OSError(errno.EPERM, "Managed target has multiple links")
        if (entry_stat.st_dev, entry_stat.st_ino) != owned_identity:
            raise OSError(errno.EAGAIN, "Managed target changed while opening")

        file = os.fdopen(file_fd, "wb", closefd=True)
        file_fd = -1
        yield file
        file.flush()
        os.fsync(file.fileno())

        opened_stat = os.fstat(file.fileno())
        final_stat = os.stat(candidate.name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened_stat.st_mode) or not stat.S_ISREG(final_stat.st_mode):
            raise OSError(errno.EPERM, "Managed target is not a regular file")
        if opened_stat.st_nlink != 1 or final_stat.st_nlink != 1:
            raise OSError(errno.EPERM, "Managed target has multiple links")
        if (final_stat.st_dev, final_stat.st_ino) != owned_identity:
            raise OSError(errno.EAGAIN, "Managed target changed while writing")
    finally:
        if file is not None and not file.closed:
            file.close()
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


@contextmanager
def _owned_file_writer_windows(root: Path, candidate: Path) -> Iterator[BinaryIO]:
    """Exclusively create the final entry as a held no-delete-share Win32 handle.

    Python has no Windows dir_fd/openat API. Holding the validated root without delete
    sharing prevents parent replacement; final-path validation binds the opened file to
    that root, and omitting delete sharing prevents entry swaps until writing completes.
    """
    import ctypes
    import msvcrt
    import ntpath
    from ctypes import wintypes

    generic_write = 0x40000000
    file_read_attributes = 0x0080
    delete = 0x00010000
    share_read = 0x00000001
    share_write = 0x00000002
    create_new = 1
    open_existing = 3
    flag_open_reparse_point = 0x00200000
    flag_backup_semantics = 0x02000000
    attribute_directory = 0x00000010
    attribute_reparse_point = 0x00000400
    file_type_disk = 0x0001
    invalid_handle = ctypes.c_void_p(-1).value

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOL)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    get_file_type = kernel32.GetFileType
    get_file_type.argtypes = [wintypes.HANDLE]
    get_file_type.restype = wintypes.DWORD
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_file_information.restype = wintypes.BOOL

    def file_info(handle):
        info = ByHandleFileInformation()
        if not get_info(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def final_path(handle) -> str:
        length = get_final_path(handle, None, 0, 0)
        if not length:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(length + 1)
        if not get_final_path(handle, buffer, len(buffer), 0):
            raise ctypes.WinError(ctypes.get_last_error())
        path = buffer.value
        if path.startswith("\\\\?\\UNC\\"):
            path = "\\\\" + path[8:]
        elif path.startswith("\\\\?\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.normpath(path))

    root_handle = create_file(
        str(root),
        file_read_attributes,
        share_read | share_write,
        None,
        open_existing,
        flag_backup_semantics | flag_open_reparse_point,
        None,
    )
    if root_handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    file_handle = invalid_handle
    file_fd = -1
    file: BinaryIO | None = None
    owned_identity: tuple[int, int, int] | None = None
    root_final: str | None = None
    complete = False
    try:
        root_info = file_info(root_handle)
        if not root_info.file_attributes & attribute_directory:
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")
        if root_info.file_attributes & attribute_reparse_point:
            raise OSError(errno.EPERM, "Managed root is a reparse point")

        file_handle = create_file(
            str(candidate),
            generic_write | file_read_attributes | delete,
            share_read,
            None,
            create_new,
            flag_open_reparse_point,
            None,
        )
        if file_handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        opened_info = file_info(file_handle)
        owned_identity = (
            opened_info.volume_serial_number,
            opened_info.file_index_high,
            opened_info.file_index_low,
        )
        if get_file_type(file_handle) != file_type_disk:
            raise OSError(errno.EPERM, "Managed target is not a disk file")
        if opened_info.file_attributes & (attribute_directory | attribute_reparse_point):
            raise OSError(errno.EPERM, "Managed target is not a direct regular file")
        if opened_info.number_of_links != 1:
            raise OSError(errno.EPERM, "Managed target has multiple links")

        root_final = final_path(root_handle)
        file_final = final_path(file_handle)
        if ntpath.dirname(file_final) != root_final:
            raise OSError(errno.EPERM, "Managed target is outside its root")
        if ntpath.basename(file_final) != ntpath.normcase(candidate.name):
            raise OSError(errno.EPERM, "Managed target name changed while opening")

        file_fd = msvcrt.open_osfhandle(
            file_handle,
            os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
        file_handle = invalid_handle
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise OSError(errno.EPERM, "Managed target is not a regular file")
        file = os.fdopen(file_fd, "wb", closefd=True)
        file_fd = -1
        yield file
        file.flush()
        os.fsync(file.fileno())
        final_handle = msvcrt.get_osfhandle(file.fileno())
        final_info = file_info(final_handle)
        final_identity = (
            final_info.volume_serial_number,
            final_info.file_index_high,
            final_info.file_index_low,
        )
        if get_file_type(final_handle) != file_type_disk:
            raise OSError(errno.EPERM, "Managed target is not a disk file")
        if final_info.file_attributes & (attribute_directory | attribute_reparse_point):
            raise OSError(errno.EPERM, "Managed target is not a direct regular file")
        if final_info.number_of_links != 1:
            raise OSError(errno.EPERM, "Managed target has multiple links")
        if final_identity != owned_identity:
            raise OSError(errno.EAGAIN, "Managed target changed while writing")
        file_final = final_path(final_handle)
        if ntpath.dirname(file_final) != root_final:
            raise OSError(errno.EPERM, "Managed target is outside its root")
        if ntpath.basename(file_final) != ntpath.normcase(candidate.name):
            raise OSError(errno.EPERM, "Managed target name changed while writing")
        complete = True
    finally:
        if not complete and owned_identity is not None:
            cleanup_handle = file_handle
            if cleanup_handle == invalid_handle:
                try:
                    cleanup_fd = file.fileno() if file is not None else file_fd
                    cleanup_handle = msvcrt.get_osfhandle(cleanup_fd)
                except (OSError, ValueError):
                    cleanup_handle = invalid_handle
            if cleanup_handle != invalid_handle:
                try:
                    cleanup_info = file_info(cleanup_handle)
                    cleanup_identity = (
                        cleanup_info.volume_serial_number,
                        cleanup_info.file_index_high,
                        cleanup_info.file_index_low,
                    )
                    cleanup_final = final_path(cleanup_handle)
                    if (
                        cleanup_identity == owned_identity
                        and root_final is not None
                        and ntpath.dirname(cleanup_final) == root_final
                        and ntpath.basename(cleanup_final) == ntpath.normcase(candidate.name)
                    ):
                        disposition = FileDispositionInfo(True)
                        set_file_information(
                            cleanup_handle,
                            4,
                            ctypes.byref(disposition),
                            ctypes.sizeof(disposition),
                        )
                except OSError:
                    pass
        if file is not None and not file.closed:
            file.close()
        if file_fd >= 0:
            os.close(file_fd)
        if file_handle != invalid_handle:
            close_handle(file_handle)
        close_handle(root_handle)


def _open_windows(root: Path, candidate: Path) -> OpenedOwnedFile:
    """Open with Win32 reparse-point checks and hold a non-replaceable file handle.

    Python does not expose a Windows dir_fd/openat equivalent. CreateFileW is the smallest
    secure platform branch: the root is held without delete sharing, reparse points are
    opened rather than followed and rejected, and the opened file's final path is checked
    against that held root before its HANDLE is converted to the streamed Python file.
    """
    import ctypes
    import msvcrt
    import ntpath
    from ctypes import wintypes

    generic_read = 0x80000000
    file_read_attributes = 0x0080
    share_read = 0x00000001
    share_write = 0x00000002
    open_existing = 3
    flag_open_reparse_point = 0x00200000
    flag_backup_semantics = 0x02000000
    attribute_directory = 0x00000010
    attribute_reparse_point = 0x00000400
    file_type_disk = 0x0001
    invalid_handle = ctypes.c_void_p(-1).value

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    get_file_type = kernel32.GetFileType
    get_file_type.argtypes = [wintypes.HANDLE]
    get_file_type.restype = wintypes.DWORD
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD

    def open_handle(path: Path, access: int, sharing: int, flags: int):
        handle = create_file(str(path), access, sharing, None, open_existing, flags, None)
        if handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def file_info(handle):
        info = ByHandleFileInformation()
        if not get_info(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def final_path(handle) -> str:
        length = get_final_path(handle, None, 0, 0)
        if not length:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(length + 1)
        if not get_final_path(handle, buffer, len(buffer), 0):
            raise ctypes.WinError(ctypes.get_last_error())
        path = buffer.value
        if path.startswith("\\\\?\\UNC\\"):
            path = "\\\\" + path[8:]
        elif path.startswith("\\\\?\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.normpath(path))

    root_handle = open_handle(
        root,
        file_read_attributes,
        share_read | share_write,
        flag_backup_semantics | flag_open_reparse_point,
    )
    file_handle = invalid_handle
    file_fd = -1
    try:
        root_info = file_info(root_handle)
        if not root_info.file_attributes & attribute_directory:
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")
        if root_info.file_attributes & attribute_reparse_point:
            raise OSError(errno.EPERM, "Managed root is a reparse point")

        file_handle = open_handle(
            candidate,
            generic_read,
            share_read,
            flag_open_reparse_point,
        )
        opened_info = file_info(file_handle)
        if get_file_type(file_handle) != file_type_disk:
            raise OSError(errno.EPERM, "Managed result is not a disk file")
        if opened_info.file_attributes & (attribute_directory | attribute_reparse_point):
            raise OSError(errno.EPERM, "Managed result is not a direct regular file")

        root_final = final_path(root_handle)
        file_final = final_path(file_handle)
        if ntpath.dirname(file_final) != root_final:
            raise OSError(errno.EPERM, "Managed result is outside its root")
        if ntpath.basename(file_final) != ntpath.normcase(candidate.name):
            raise OSError(errno.EPERM, "Managed result name changed while opening")

        file_fd = msvcrt.open_osfhandle(
            file_handle,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        file_handle = invalid_handle
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError(errno.EPERM, "Managed result is not a regular file")
        file = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = -1
        return OpenedOwnedFile(file=file, path=candidate, stat_result=opened_stat)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if file_handle != invalid_handle:
            close_handle(file_handle)
        close_handle(root_handle)
