"""Open and stream managed files without reopening a validated pathname."""

from __future__ import annotations

import errno
import os
import secrets
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import anyio
from fastapi import HTTPException
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send


@dataclass(frozen=True)
class OpenedOwnedFile:
    file: BinaryIO
    path: Path
    stat_result: os.stat_result


class HeldFileResponse(FileResponse):
    """Stream one validated open file and close it on every ASGI exit path."""

    def __init__(
        self,
        opened: OpenedOwnedFile,
        *,
        filename: str,
        media_type: str | None = None,
    ) -> None:
        self.file = opened.file
        try:
            super().__init__(
                opened.path,
                filename=filename,
                media_type=media_type,
                stat_result=opened.stat_result,
            )
        except BaseException:
            self.close()
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # The response-level finally also covers normal completion and disconnects.
            self.close()

    async def _handle_simple(
        self, send: Send, send_header_only: bool, send_pathsend: bool
    ) -> None:
        del send_pathsend  # A validated held handle must never fall back to pathsend.
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

    async def _handle_single_range(
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

    async def _handle_multiple_ranges(
        self,
        send: Send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        boundary = secrets.token_hex(13)
        content_length, header_generator = self.generate_multipart(
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
    """Yield a secure temporary file and atomically publish it as a direct child."""
    root_path, candidate_path = _direct_child_paths(root, value)
    if os.name == "nt":
        with _atomic_owned_file_windows(root_path, candidate_path) as file:
            yield file
    else:
        with _atomic_owned_file_posix(root_path, candidate_path) as file:
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


def _validate_posix_target(root_fd: int, name: str) -> None:
    try:
        target_stat = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target_stat.st_mode):
        raise OSError(errno.EPERM, "Managed target is not a regular file")


@contextmanager
def _atomic_owned_file_posix(root: Path, candidate: Path) -> Iterator[BinaryIO]:
    """Create and publish through one held directory without following links."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError(errno.ENOTSUP, "Secure no-follow writes are unavailable")

    common_flags = getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | common_flags)
    temp_name: str | None = None
    temp_fd = -1
    file: BinaryIO | None = None
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")
        _validate_posix_target(root_fd, candidate.name)

        for _ in range(128):
            temp_name = f".preview-{secrets.token_hex(12)}.tmp"
            try:
                temp_fd = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | common_flags,
                    0o600,
                    dir_fd=root_fd,
                )
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(errno.EEXIST, "Could not reserve a preview file")

        file = os.fdopen(temp_fd, "wb", closefd=True)
        temp_fd = -1
        yield file
        file.flush()
        os.fsync(file.fileno())

        opened_stat = os.fstat(file.fileno())
        entry_stat = os.stat(temp_name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry_stat.st_mode) or (
            entry_stat.st_dev,
            entry_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise OSError(errno.EAGAIN, "Managed temporary file changed before publish")

        _validate_posix_target(root_fd, candidate.name)
        os.replace(temp_name, candidate.name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        temp_name = None
        os.fsync(root_fd)
    finally:
        if file is not None and not file.closed:
            file.close()
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        os.close(root_fd)


def _validate_windows_target(candidate: Path) -> None:
    try:
        target_stat = os.lstat(candidate)
    except FileNotFoundError:
        return
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
    attributes = getattr(target_stat, "st_file_attributes", 0)
    if attributes & reparse_point or not stat.S_ISREG(target_stat.st_mode):
        raise OSError(errno.EPERM, "Managed target is not a regular file")


@contextmanager
def _atomic_owned_file_windows(root: Path, candidate: Path) -> Iterator[BinaryIO]:
    """Hold a non-replaceable root, write a unique file, then replace its direct child.

    Python has no Windows dir_fd/openat API. A no-delete-sharing CreateFileW root handle
    prevents the validated directory from being swapped; mkstemp uses CREATE_NEW, and
    os.replace atomically replaces the destination entry rather than following reparse data.
    """
    import ctypes
    from ctypes import wintypes

    file_read_attributes = 0x0080
    share_read = 0x00000001
    share_write = 0x00000002
    open_existing = 3
    flag_open_reparse_point = 0x00200000
    flag_backup_semantics = 0x02000000
    attribute_directory = 0x00000010
    attribute_reparse_point = 0x00000400
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
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation)]
    get_info.restype = wintypes.BOOL

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

    temp_path: str | None = None
    temp_fd = -1
    file: BinaryIO | None = None
    try:
        root_info = ByHandleFileInformation()
        if not get_info(root_handle, ctypes.byref(root_info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not root_info.file_attributes & attribute_directory:
            raise OSError(errno.ENOTDIR, "Managed root is not a directory")
        if root_info.file_attributes & attribute_reparse_point:
            raise OSError(errno.EPERM, "Managed root is a reparse point")
        _validate_windows_target(candidate)

        temp_fd, temp_path = tempfile.mkstemp(prefix=".preview-", suffix=".tmp", dir=root)
        file = os.fdopen(temp_fd, "wb", closefd=True)
        temp_fd = -1
        yield file
        file.flush()
        os.fsync(file.fileno())
        opened_stat = os.fstat(file.fileno())
        file.close()

        entry_stat = os.lstat(temp_path)
        if not stat.S_ISREG(entry_stat.st_mode) or (
            entry_stat.st_dev,
            entry_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise OSError(errno.EAGAIN, "Managed temporary file changed before publish")

        _validate_windows_target(candidate)
        os.replace(temp_path, candidate)
        temp_path = None
    finally:
        if file is not None and not file.closed:
            file.close()
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
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
