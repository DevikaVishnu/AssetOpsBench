"""Read-only observation of deepagents' framework file operations.

deepagents moves information through files in graph state: oversized tool
results are evicted to ``/large_tool_results/<id>``, sub-agents hand files back
through the ``task`` result, and agents retrieve them with ``read_file``.  None
of that reaches LangChain callbacks.  :class:`ObservedBackend` wraps the run's
``StateBackend`` and reports each write, edit and read to the capture handler
(:meth:`SubagentCaptureHandler.record_file_op`).

Attribution comes from the backend runtime's ``tool_call_id`` (one
``ToolRuntime`` per tool call).  deepagents 0.5.3 only passes that runtime to a
*callable* backend, which it deprecates (removal planned for v0.7) and warns
about on every resolution.  Backend resolutions made outside a tool call
(model-call middleware, summarization) have no ``tool_call_id`` and are
recorded as ``"unattributed"``; nothing is inferred from paths or timing.

Event semantics:
  * ``file_write`` / ``file_edit`` record a *queued* ``files`` channel update
    (``StateBackend`` sends it; LangGraph applies it at the node boundary),
    not proof that the update was committed.
  * ``file_read`` hashes the raw window the backend returned, before
    ``read_file``'s line-number formatting and truncation.
  * Storage is not delivery: these events say what was in files, never what a
    model received.

The wrapper never changes behavior: every call is forwarded to the shared inner
backend, its result object is returned unchanged and its exceptions propagate
unchanged.  Only the bookkeeping after the call is guarded.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

_log = logging.getLogger(__name__)


def _sha256(text: Any) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None


def _size(text: Any) -> int | None:
    return len(text) if isinstance(text, str) else None


class ObservedBackend(BackendProtocol):
    """Forwards every ``BackendProtocol`` method to *inner* and reports
    writes, edits and reads to *sink*.

    Subclasses ``BackendProtocol`` (not ``SandboxBackendProtocol``) so
    ``supports_execution`` and ``artifacts_root`` resolve exactly as for a
    plain ``StateBackend``.
    """

    def __init__(self, inner: BackendProtocol, sink: Any, tool_call_id: str | None) -> None:
        self._inner = inner
        self._sink = sink
        self._tool_call_id = tool_call_id

    # -- recording (never raises) ------------------------------------------------

    def _record(self, type_: str, build: Callable[[], dict[str, Any]]) -> None:
        try:
            self._sink.record_file_op(
                type_,
                tool_call_id=self._tool_call_id,
                attribution="tool_call" if self._tool_call_id else "unattributed",
                **build(),
            )
        except Exception:
            _log.exception("file observer: recording %s failed", type_)

    def _record_write(self, file_path: str, content: str, result: WriteResult) -> None:
        self._record(
            "file_write",
            lambda: {
                "path": file_path,
                "sha256": _sha256(content),
                "size": _size(content),
                "error": result.error,
                "result_path": result.path,
            },
        )

    def _record_edit(self, file_path: str, old: str, new: str, replace_all: bool, result: EditResult) -> None:
        self._record(
            "file_edit",
            lambda: {
                "path": file_path,
                "old_sha256": _sha256(old),
                "new_sha256": _sha256(new),
                "replace_all": replace_all,
                "error": result.error,
                "occurrences": result.occurrences,
            },
        )

    def _record_read(self, file_path: str, offset: int, limit: int, result: ReadResult) -> None:
        def build() -> dict[str, Any]:
            fd = result.file_data or {}
            window = fd.get("content")
            return {
                "path": file_path,
                "offset": offset,
                "limit": limit,
                "error": result.error,
                "encoding": fd.get("encoding"),
                "created_at": fd.get("created_at"),
                "modified_at": fd.get("modified_at"),
                "window_sha256": _sha256(window),
                "window_size": _size(window),
            }

        self._record("file_read", build)

    # -- recorded operations -----------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        result = self._inner.write(file_path, content)
        self._record_write(file_path, content, result)
        return result

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        result = await self._inner.awrite(file_path, content)
        self._record_write(file_path, content, result)
        return result

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        result = self._inner.edit(file_path, old_string, new_string, replace_all)
        self._record_edit(file_path, old_string, new_string, replace_all, result)
        return result

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        result = await self._inner.aedit(file_path, old_string, new_string, replace_all)
        self._record_edit(file_path, old_string, new_string, replace_all, result)
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        result = self._inner.read(file_path, offset, limit)
        self._record_read(file_path, offset, limit, result)
        return result

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        result = await self._inner.aread(file_path, offset, limit)
        self._record_read(file_path, offset, limit, result)
        return result

    # -- forwarded only ----------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        return self._inner.ls(path)

    async def als(self, path: str) -> LsResult:
        return await self._inner.als(path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self._inner.grep(pattern, path, glob)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return await self._inner.agrep(pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return self._inner.glob(pattern, path)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        return await self._inner.aglob(pattern, path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._inner.upload_files(files)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return await self._inner.aupload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._inner.download_files(paths)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await self._inner.adownload_files(paths)

    def ls_info(self, path: str) -> list[FileInfo]:
        return self._inner.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        return await self._inner.als_info(path)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self._inner.glob_info(pattern, path)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await self._inner.aglob_info(pattern, path)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        return self._inner.grep_raw(pattern, path, glob)

    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        return await self._inner.agrep_raw(pattern, path, glob)


def observing_backend_factory(inner: BackendProtocol, sink: Any) -> Callable[[Any], BackendProtocol]:
    """Backend factory for ``create_deep_agent(backend=...)``.

    deepagents calls it once per backend resolution; a tool call's
    ``ToolRuntime`` carries that call's ``tool_call_id``.  All resolutions
    share *inner*, so file state is identical to passing *inner* directly.
    """

    def factory(runtime: Any) -> BackendProtocol:
        return ObservedBackend(inner, sink, getattr(runtime, "tool_call_id", None))

    return factory
