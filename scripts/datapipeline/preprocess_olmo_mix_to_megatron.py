#!/usr/bin/env python3
"""Stream an OLMo-Mix subset into Megatron-LM IndexedDataset files.

Expected input layout under --data-dir:

    data/
      dclm/**/*.jsonl.zstd
      pes2o/**/*.json.gz
      starcoder/**/*.json.gz
      wiki/**/*.json.gz

Only files that actually exist are processed. Missing shard numbers are normal: the
script never assumes that the original OLMo-Mix download is complete.

The output for each source/task is a sequence of Megatron prefixes such as:

    <output>/dclm/task-000-of-001/
      dclm-t000-p00000_text_document.bin
      dclm-t000-p00000_text_document.idx

Pass the prefix ending in ``_text_document`` (without .bin/.idx) to Megatron.

This script intentionally:
  * does not split sentences;
  * does not normalize or strip document text;
  * appends exactly one tokenizer.eod to every non-empty document;
  * preserves Megatron's HuggingFace tokenization semantics and IndexedDataset format;
  * reads gzip/zstd files directly without materializing decompressed JSONL;
  * rolls output only at document boundaries;
  * supports deterministic, size-balanced file assignment across independent tasks;
  * records manifests and uses .done markers for safe resume.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import gzip
import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from array import array
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)


# Worker globals are initialized once per process.
_WORKER_TOKENIZER: Any = None
_WORKER_EOD: Optional[int] = None

SOURCE_ORDER = ("dclm", "pes2o", "starcoder", "wiki")
SOURCE_DIRS = {
    "dclm": "dclm",
    "pes2o": "pes2o",
    "starcoder": "starcoder",
    "wiki": "wiki",
}
TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "tokenizer_metadata.json",
)
PIPELINE_SEMANTICS_VERSION = 1
SEMANTIC_CONFIG_FIELDS = (
    "pipeline_semantics_version",
    "source",
    "task_id",
    "num_tasks",
    "input_manifest_sha256",
    "tokenizer_fingerprint_sha256",
    "vocab_size",
    "eod_id",
    "dtype",
    "use_fast_tokenizer",
    "max_bin_bytes",
    "max_documents",
    "sentence_splitting",
    "append_eod",
    "error_policy",
)


@dataclass(frozen=True)
class InputFile:
    index: int
    path: Path
    relative_path: str
    size_bytes: int


@dataclass
class Cursor:
    file_index: int
    relative_path: str
    line_number: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "file_index": self.file_index,
            "relative_path": self.relative_path,
            "line_number": self.line_number,
        }


@dataclass
class EncodeResult:
    ok: bool
    cursor: Cursor
    raw_bytes: int
    char_count: int = 0
    # array('I') keeps worker-to-parent IPC at four bytes per token instead of
    # pickling a Python list whose integer objects can consume roughly 8x as much.
    token_ids: Optional[array] = None
    error_kind: Optional[str] = None
    error_message: Optional[str] = None


class WorkerBatchTimeoutError(RuntimeError):
    pass


class PartWriter:
    """Own one temporary Megatron IndexedDataset and atomically publish it."""

    def __init__(
        self,
        *,
        indexed_dataset_module: Any,
        output_dir: Path,
        source: str,
        task_id: int,
        part_id: int,
        dtype: Any,
        dtype_itemsize: int,
        eod_id: int,
        manifest_sha256: str,
        tokenizer_metadata: Dict[str, Any],
        run_config: Dict[str, Any],
    ) -> None:
        self.indexed_dataset = indexed_dataset_module
        self.output_dir = output_dir
        self.source = source
        self.task_id = task_id
        self.part_id = part_id
        self.dtype = dtype
        self.dtype_itemsize = dtype_itemsize
        self.eod_id = eod_id
        self.manifest_sha256 = manifest_sha256
        self.tokenizer_metadata = tokenizer_metadata
        self.run_config = run_config

        stem = f"{source}-t{task_id:03d}-p{part_id:05d}_text_document"
        self.final_prefix = output_dir / stem
        self.temp_prefix = output_dir / f"{stem}.tmp"

        self.temp_bin = Path(f"{self.temp_prefix}.bin")
        self.temp_idx = Path(f"{self.temp_prefix}.idx")
        self.final_bin = Path(f"{self.final_prefix}.bin")
        self.final_idx = Path(f"{self.final_prefix}.idx")
        self.meta_path = output_dir / f"{stem}.meta.json"
        self.meta_tmp = output_dir / f"{stem}.meta.json.tmp"
        self.done_path = output_dir / f"{stem}.done"
        self.done_tmp = output_dir / f"{stem}.done.tmp"

        for path in (self.temp_bin, self.temp_idx, self.meta_tmp, self.done_tmp):
            path.unlink(missing_ok=True)

        self.builder = self.indexed_dataset.IndexedDatasetBuilder(
            str(self.temp_bin), dtype=self.dtype
        )
        self.started_at = utc_now()
        self.documents = 0
        self.tokens = 0
        self.raw_lines = 0
        self.raw_bytes = 0
        self.bad_records = 0
        self.bad_counts: Counter[str] = Counter()
        self.max_document_tokens = 0
        self.max_document_characters = 0
        self.first_cursor: Optional[Cursor] = None
        self.last_cursor: Optional[Cursor] = None
        self.file_spans: Dict[int, Dict[str, Any]] = {}

    def observe_line(self, result: EncodeResult) -> None:
        """Count every consumed input line, including rejected records."""
        self.raw_lines += 1
        self.raw_bytes += result.raw_bytes
        if self.first_cursor is None:
            self.first_cursor = result.cursor
        self.last_cursor = result.cursor

        span = self.file_spans.get(result.cursor.file_index)
        if span is None:
            span = {
                "file_index": result.cursor.file_index,
                "relative_path": result.cursor.relative_path,
                "first_line": result.cursor.line_number,
                "last_line": result.cursor.line_number,
                "raw_lines": 0,
                "documents": 0,
                "bad_records": 0,
            }
            self.file_spans[result.cursor.file_index] = span
        span["last_line"] = result.cursor.line_number
        span["raw_lines"] += 1

        if result.ok:
            span["documents"] += 1
        else:
            span["bad_records"] += 1
            self.bad_records += 1
            self.bad_counts[result.error_kind or "unknown"] += 1

    def add_document(self, result: EncodeResult) -> None:
        assert result.ok
        assert result.token_ids is not None
        token_ids = result.token_ids
        assert token_ids, "empty token list must not be written"
        assert token_ids[-1] == self.eod_id, "document must end with EOD"

        # No sentence splitting: exactly one sequence length per document.
        self.builder.add_document(token_ids, [len(token_ids)])
        self.documents += 1
        self.tokens += len(token_ids)
        self.max_document_tokens = max(self.max_document_tokens, len(token_ids))
        self.max_document_characters = max(self.max_document_characters, result.char_count)

    @property
    def bin_bytes(self) -> int:
        return self.tokens * self.dtype_itemsize

    def finalize(self) -> Path:
        if self.documents <= 0:
            raise RuntimeError("refusing to finalize an empty IndexedDataset")
        if self.last_cursor is None:
            raise RuntimeError("part has documents but no input cursor")

        self.builder.finalize(str(self.temp_idx))

        actual_temp_bin_bytes = self.temp_bin.stat().st_size
        expected_bin_bytes = self.bin_bytes
        if actual_temp_bin_bytes != expected_bin_bytes:
            raise RuntimeError(
                f"bin size mismatch for {self.temp_bin}: "
                f"actual={actual_temp_bin_bytes}, expected={expected_bin_bytes}"
            )

        # Re-open through the exact Megatron reader before publishing the files.
        dataset = self.indexed_dataset.IndexedDataset(str(self.temp_prefix), mmap=True)
        try:
            document_count = int(dataset.document_indices.shape[0] - 1)
            sequence_count = len(dataset)
            token_count = int(dataset.sequence_lengths.sum())
            if document_count != self.documents:
                raise RuntimeError(
                    f"document count mismatch: index={document_count}, expected={self.documents}"
                )
            # With sentence splitting disabled, one document is one sequence.
            if sequence_count != self.documents:
                raise RuntimeError(
                    f"sequence count mismatch: index={sequence_count}, expected={self.documents}"
                )
            if token_count != self.tokens:
                raise RuntimeError(
                    f"token count mismatch: index={token_count}, expected={self.tokens}"
                )

            sample_indices = sorted({0, sequence_count // 2, sequence_count - 1})
            for sample_index in sample_indices:
                sample = dataset[sample_index]
                if len(sample) == 0 or int(sample[-1]) != self.eod_id:
                    raise RuntimeError(
                        f"sample {sample_index} in {self.temp_prefix} does not end with EOD"
                    )
        finally:
            del dataset
            gc.collect()

        meta = {
            "format": "Megatron-LM IndexedDataset",
            "level": "document",
            "sentence_splitting": False,
            "append_eod": True,
            "source": self.source,
            "task_id": self.task_id,
            "part_id": self.part_id,
            "output_prefix": str(self.final_prefix.resolve()),
            "bin_file": self.final_bin.name,
            "idx_file": self.final_idx.name,
            "documents": self.documents,
            "sequences": self.documents,
            "tokens": self.tokens,
            "bin_bytes": expected_bin_bytes,
            "dtype": self.dtype.__name__,
            "dtype_itemsize": self.dtype_itemsize,
            "eod_id": self.eod_id,
            "raw_lines": self.raw_lines,
            "raw_bytes": self.raw_bytes,
            "bad_records": self.bad_records,
            "bad_counts": dict(sorted(self.bad_counts.items())),
            "max_document_tokens": self.max_document_tokens,
            "max_document_characters": self.max_document_characters,
            "first_input_cursor": self.first_cursor.as_dict() if self.first_cursor else None,
            "last_input_cursor": self.last_cursor.as_dict(),
            "input_spans": [self.file_spans[k] for k in sorted(self.file_spans)],
            "input_manifest_sha256": self.manifest_sha256,
            "tokenizer": self.tokenizer_metadata,
            "run_config": self.run_config,
            "started_at_utc": self.started_at,
            "finished_at_utc": utc_now(),
        }
        write_json_atomic(self.meta_tmp, self.meta_path, meta)

        # Rename on the same filesystem. The .done marker is published last.
        os.replace(self.temp_bin, self.final_bin)
        os.replace(self.temp_idx, self.final_idx)
        write_text_atomic(self.done_tmp, self.done_path, json.dumps({"ok": True}) + "\n")
        return self.final_prefix


class PendingPartStats:
    """Hold rejected lines that appear before the next valid document opens a builder."""

    def __init__(self) -> None:
        self.results: List[EncodeResult] = []

    def add(self, result: EncodeResult) -> None:
        self.results.append(result)

    def transfer_to(self, writer: PartWriter) -> None:
        for result in self.results:
            writer.observe_line(result)
        self.results.clear()

    def clear(self) -> None:
        self.results.clear()


@contextlib.contextmanager
def open_compressed_binary(path: Path) -> Iterator[BinaryIO]:
    name = path.name.lower()
    if name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            yield stream
        return

    if name.endswith((".zstd", ".zst")):
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "zstandard is required for .zstd/.zst input; install it with: pip install zstandard"
            ) from exc
        with path.open("rb") as raw:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                with io.BufferedReader(reader, buffer_size=8 * 1024 * 1024) as buffered:
                    yield buffered
        return

    raise ValueError(f"unsupported compressed input: {path}")


def _initialize_worker(
    tokenizer_model: str,
    use_fast_tokenizer: bool,
    expected_vocab_size: int,
    expected_eod: int,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOD
    configure_process_environment()
    # Tokenizer workers never run a model. Prevent Transformers from importing
    # PyTorch (~400-600 MiB RSS per spawned process in the target environment).
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # Importing megatron.core in every spawn worker also imports PyTorch and the
    # CUDA inference stack (including unified-memory initialization). Megatron's
    # Hugging Face wrapper tokenizes with exactly the two calls in _encode_one,
    # so workers only need the underlying lightweight tokenizer.
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_model,
        use_fast=use_fast_tokenizer,
        local_files_only=True,
    )
    actual_vocab_size = len(_WORKER_TOKENIZER)
    _WORKER_EOD = _WORKER_TOKENIZER.eos_token_id
    if _WORKER_EOD is None:
        raise RuntimeError("Hugging Face tokenizer returned eos_token_id=None")
    _WORKER_EOD = int(_WORKER_EOD)
    if actual_vocab_size != expected_vocab_size or _WORKER_EOD != expected_eod:
        raise RuntimeError(
            "worker tokenizer disagrees with parent validation: "
            f"vocab={actual_vocab_size}/{expected_vocab_size}, "
            f"eod={_WORKER_EOD}/{expected_eod}"
        )


def _encode_one(item: Tuple[int, str, int, bytes]) -> EncodeResult:
    file_index, relative_path, line_number, raw_line = item
    cursor = Cursor(file_index, relative_path, line_number)
    raw_bytes = len(raw_line)

    try:
        record = json.loads(raw_line)
    except Exception as exc:  # malformed line is a skippable data error
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            error_kind="json_decode_error",
            error_message=limited_repr(exc),
        )

    if not isinstance(record, dict):
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            error_kind="record_not_object",
            error_message=f"got {type(record).__name__}",
        )

    if "text" not in record:
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            error_kind="missing_text",
        )

    text = record["text"]
    if not isinstance(text, str):
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            error_kind="text_not_string",
            error_message=f"got {type(text).__name__}",
        )

    # This check does not alter text; the original string is sent to the tokenizer.
    if text == "" or text.isspace():
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            char_count=len(text),
            error_kind="empty_text",
        )

    try:
        tokens = _WORKER_TOKENIZER.tokenize(text)
        token_ids = _WORKER_TOKENIZER.convert_tokens_to_ids(tokens)
    except Exception as exc:
        # Tokenizer failures are marked separately so the main process can fail fast.
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            char_count=len(text),
            error_kind="tokenizer_error",
            error_message=limited_repr(exc),
        )

    if not token_ids:
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            char_count=len(text),
            error_kind="zero_tokens",
        )

    token_ids.append(_WORKER_EOD)
    try:
        compact_token_ids = array("I", token_ids)
    except (OverflowError, TypeError, ValueError) as exc:
        return EncodeResult(
            ok=False,
            cursor=cursor,
            raw_bytes=raw_bytes,
            char_count=len(text),
            error_kind="tokenizer_error",
            error_message=f"token ids cannot be represented as uint32: {limited_repr(exc)}",
        )
    if compact_token_ids.itemsize != 4:
        raise RuntimeError(
            f"array('I') has unexpected itemsize={compact_token_ids.itemsize}; expected 4"
        )
    return EncodeResult(
        ok=True,
        cursor=cursor,
        raw_bytes=raw_bytes,
        char_count=len(text),
        token_ids=compact_token_ids,
    )


def _encode_batch(items: Sequence[Tuple[int, str, int, bytes]]) -> List[EncodeResult]:
    return [_encode_one(item) for item in items]


def _take_batch(iterator: Iterator[Any], batch_size: int) -> List[Any]:
    batch = []
    for _ in range(batch_size):
        try:
            batch.append(next(iterator))
        except StopIteration:
            break
    return batch


def iter_bounded_ordered_results(
    pool: Any,
    items: Iterable[Tuple[int, str, int, bytes]],
    *,
    batch_size: int,
    max_inflight_batches: int,
    heartbeat_seconds: float,
    batch_timeout_seconds: float,
    on_wait: Optional[Callable[[int, float], None]] = None,
) -> Iterator[Tuple[EncodeResult, int]]:
    """Tokenize with bounded memory while preserving exact input order.

    multiprocessing.Pool.imap may retain an unbounded number of completed,
    out-of-order results behind one slow input. Here only
    ``max_inflight_batches * batch_size`` input records can be submitted at once.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_inflight_batches < 1:
        raise ValueError("max_inflight_batches must be >= 1")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be > 0")
    if batch_timeout_seconds < 0:
        raise ValueError("batch_timeout_seconds must be >= 0")

    item_iterator = iter(items)
    pending: Deque[Any] = deque()
    exhausted = False

    def fill_pending() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < max_inflight_batches:
            batch = _take_batch(item_iterator, batch_size)
            if not batch:
                exhausted = True
                break
            pending.append(pool.apply_async(_encode_batch, (batch,)))

    fill_pending()
    while pending:
        oldest = pending.popleft()
        wait_started = time.monotonic()
        while True:
            try:
                batch_results = oldest.get(timeout=heartbeat_seconds)
                break
            except mp.TimeoutError:
                waited = time.monotonic() - wait_started
                if on_wait is not None:
                    on_wait(len(pending) + 1, waited)
                if batch_timeout_seconds and waited >= batch_timeout_seconds:
                    raise WorkerBatchTimeoutError(
                        "oldest tokenization batch produced no result for "
                        f"{waited:.0f}s; a worker may have exited during the task"
                    )

        # Include the batch currently held by the parent in the bound. Refill
        # only after it has been consumed so result memory never exceeds the
        # configured number of batches.
        inflight_batches = len(pending) + 1
        for result in batch_results:
            yield result, inflight_batches
        del batch_results
        fill_pending()


def configure_process_environment() -> None:
    # Avoid nested per-worker thread pools and accidental network access.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def limited_repr(exc: BaseException, limit: int = 500) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer_model: Path) -> Dict[str, Any]:
    files = []
    for name in TOKENIZER_FINGERPRINT_FILES:
        candidate = tokenizer_model / name if tokenizer_model.is_dir() else tokenizer_model
        if candidate.exists() and candidate.is_file():
            files.append(
                {
                    "name": candidate.name,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
        if tokenizer_model.is_file():
            break
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(tokenizer_model.resolve()),
        "files": files,
        "fingerprint_sha256": digest,
    }


def find_megatron_root(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        expected = candidate / "megatron" / "core" / "datasets" / "indexed_dataset.py"
        if expected.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the Megatron-LM repository root. Run from the repository root, "
        "place this script inside the repository, or pass --megatron-root."
    )


def insert_megatron_root(root: Path) -> None:
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def git_metadata(root: Path) -> Dict[str, Any]:
    def run_git(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git("status", "--short")
    return {
        "root": str(root.resolve()),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run_git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def discover_source_files(data_dir: Path, source: str) -> List[Path]:
    root = data_dir / SOURCE_DIRS[source]
    if not root.is_dir():
        return []

    if source == "dclm":
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.lower().endswith((".jsonl.zstd", ".jsonl.zst"))
        ]
    else:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.lower().endswith(".json.gz")
        ]
    return sorted(path.resolve() for path in files)


def assign_files_by_size(
    paths: Sequence[Path], data_dir: Path, num_tasks: int, task_id: int
) -> Tuple[List[InputFile], List[int]]:
    if num_tasks < 1:
        raise ValueError("--num-tasks must be >= 1")
    if not 0 <= task_id < num_tasks:
        raise ValueError("--task-id must satisfy 0 <= task-id < num-tasks")

    entries = [
        {
            "path": path,
            "relative_path": path.relative_to(data_dir).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]

    # Largest-first greedy bin packing gives deterministic and reasonably balanced tasks.
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_tasks)]
    totals = [0 for _ in range(num_tasks)]
    for entry in sorted(entries, key=lambda item: (-item["size_bytes"], item["relative_path"])):
        target = min(range(num_tasks), key=lambda idx: (totals[idx], idx))
        bins[target].append(entry)
        totals[target] += entry["size_bytes"]

    selected = sorted(bins[task_id], key=lambda item: item["relative_path"])
    files = [
        InputFile(
            index=index,
            path=item["path"],
            relative_path=item["relative_path"],
            size_bytes=item["size_bytes"],
        )
        for index, item in enumerate(selected)
    ]
    return files, totals


def manifest_payload(files: Sequence[InputFile]) -> Tuple[List[Dict[str, Any]], str]:
    rows = [
        {
            "file_index": item.index,
            "relative_path": item.relative_path,
            "compressed_bytes": item.size_bytes,
        }
        for item in files
    ]
    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    return rows, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_json_atomic(temp: Path, final: Path, payload: Dict[str, Any]) -> None:
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, final)


def write_text_atomic(temp: Path, final: Path, text: str) -> None:
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, final)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def config_digest(config: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_semantic_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resume-critical fields and apply documented legacy defaults."""
    normalized = {field: config.get(field) for field in SEMANTIC_CONFIG_FIELDS}
    # Runs created before --use-fast-tokenizer existed always used the slow
    # implementation. Treating a missing field as False prevents mixed output.
    if "use_fast_tokenizer" not in config:
        normalized["use_fast_tokenizer"] = False
    # Version 1 describes both the original pipeline and the current transport-
    # only memory changes; token IDs and document boundaries are unchanged.
    if "pipeline_semantics_version" not in config:
        normalized["pipeline_semantics_version"] = 1
    return normalized


def semantic_config_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("semantic_config", metadata.get("stable_config"))
    if not isinstance(raw, dict):
        raise RuntimeError(
            "existing run_metadata.json has no semantic_config/stable_config object"
        )
    return normalize_semantic_config(raw)


def semantic_config_mismatches(
    existing: Dict[str, Any], current: Dict[str, Any]
) -> List[Tuple[str, Any, Any]]:
    return [
        (field, existing.get(field), current.get(field))
        for field in SEMANTIC_CONFIG_FIELDS
        if existing.get(field) != current.get(field)
    ]


def resolve_tokenizer_mode(
    requested_use_fast: Optional[bool], run_metadata_path: Path
) -> Tuple[bool, str]:
    """Choose fast for new output and preserve the original mode on resume."""
    if requested_use_fast is not None:
        return requested_use_fast, "explicit command-line option"
    if not run_metadata_path.exists():
        return True, "new-output default"

    metadata = load_json(run_metadata_path)
    existing_config = semantic_config_from_metadata(metadata)
    existing_mode = existing_config["use_fast_tokenizer"]
    if not isinstance(existing_mode, bool):
        raise RuntimeError(
            "existing run_metadata.json has an invalid use_fast_tokenizer value: "
            f"{existing_mode!r}"
        )
    return existing_mode, "existing output metadata"


def existing_tokenizer_mode(run_metadata_path: Path) -> Optional[bool]:
    if not run_metadata_path.exists():
        return None
    metadata = load_json(run_metadata_path)
    mode = semantic_config_from_metadata(metadata)["use_fast_tokenizer"]
    if not isinstance(mode, bool):
        raise RuntimeError(
            f"{run_metadata_path} has an invalid use_fast_tokenizer value: {mode!r}"
        )
    return mode


def shared_tokenizer_mode_for_all_sources(
    output_dir: Path, task_id: int, num_tasks: int
) -> Optional[bool]:
    modes: Dict[bool, List[Path]] = defaultdict(list)
    for source in SOURCE_ORDER:
        metadata_path = (
            output_dir
            / source
            / f"task-{task_id:03d}-of-{num_tasks:03d}"
            / "run_metadata.json"
        )
        mode = existing_tokenizer_mode(metadata_path)
        if mode is not None:
            modes[mode].append(metadata_path)
    if len(modes) > 1:
        details = "\n".join(
            f"  - {'fast' if mode else 'slow'}: "
            + ", ".join(str(path.parent) for path in paths)
            for mode, paths in sorted(modes.items())
        )
        raise RuntimeError(
            "--source all found existing outputs with conflicting tokenizer modes:\n"
            f"{details}\nProcess the sources separately after verifying their history."
        )
    return next(iter(modes), None)


def _read_int_file(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def process_rss_bytes(pid: int) -> Optional[int]:
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _cgroup_relative_paths() -> Tuple[Optional[str], Optional[str]]:
    v2_path: Optional[str] = None
    v1_memory_path: Optional[str] = None
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2_path, v1_memory_path
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative_path = parts
        if not controllers:
            v2_path = relative_path
        elif "memory" in controllers.split(","):
            v1_memory_path = relative_path
    return v2_path, v1_memory_path


def cgroup_memory_bytes() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return current, limit, and peak bytes for cgroup v2 or v1."""
    v2_relative, v1_relative = _cgroup_relative_paths()
    candidates: List[Tuple[Path, str, str, str]] = []

    v2_root = Path("/sys/fs/cgroup")
    if v2_relative is not None:
        candidates.append(
            (
                v2_root / v2_relative.lstrip("/"),
                "memory.current",
                "memory.max",
                "memory.peak",
            )
        )
    candidates.append((v2_root, "memory.current", "memory.max", "memory.peak"))

    v1_root = Path("/sys/fs/cgroup/memory")
    if v1_relative is not None:
        candidates.append(
            (
                v1_root / v1_relative.lstrip("/"),
                "memory.usage_in_bytes",
                "memory.limit_in_bytes",
                "memory.max_usage_in_bytes",
            )
        )
    candidates.append(
        (
            v1_root,
            "memory.usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.max_usage_in_bytes",
        )
    )

    seen = set()
    for base, current_name, limit_name, peak_name in candidates:
        if base in seen:
            continue
        seen.add(base)
        current_path = base / current_name
        if not current_path.is_file():
            continue
        current = _read_int_file(current_path)
        limit = _read_int_file(base / limit_name)
        peak = _read_int_file(base / peak_name)
        # cgroup v1 represents an unlimited hierarchy with a huge sentinel.
        if limit is not None and limit >= (1 << 60):
            limit = None
        return current, limit, peak
    return None, None, None


def cgroup_oom_status() -> str:
    v2_relative, v1_relative = _cgroup_relative_paths()
    v2_bases = [Path("/sys/fs/cgroup")]
    if v2_relative is not None:
        v2_bases.insert(0, v2_bases[0] / v2_relative.lstrip("/"))
    for base in dict.fromkeys(v2_bases):
        events_path = base / "memory.events"
        try:
            rows = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        events = {}
        for row in rows:
            fields = row.split()
            if len(fields) == 2:
                try:
                    events[fields[0]] = int(fields[1])
                except ValueError:
                    pass
        return f"oom={events.get('oom', 0)},oom_kill={events.get('oom_kill', 0)}"

    v1_bases = [Path("/sys/fs/cgroup/memory")]
    if v1_relative is not None:
        v1_bases.insert(0, v1_bases[0] / v1_relative.lstrip("/"))
    for base in dict.fromkeys(v1_bases):
        failcnt_path = base / "memory.failcnt"
        if failcnt_path.is_file():
            failcnt = _read_int_file(failcnt_path)
            return f"failcnt={failcnt if failcnt is not None else 'unknown'}"
    return "unavailable"


def pool_memory_status(pool: Any) -> str:
    main_rss = process_rss_bytes(os.getpid())
    worker_rss = []
    for process in getattr(pool, "_pool", ()):  # Pool has no public worker list.
        pid = getattr(process, "pid", None)
        if pid is not None:
            rss = process_rss_bytes(pid)
            if rss is not None:
                worker_rss.append(rss)

    def display(value: Optional[int]) -> str:
        return human_bytes(value) if value is not None else "unknown"

    worker_total = sum(worker_rss) if worker_rss else None
    worker_max = max(worker_rss) if worker_rss else None
    cgroup_current, cgroup_limit, cgroup_peak = cgroup_memory_bytes()
    return (
        f"rss(main/workers/max)={display(main_rss)}/{display(worker_total)}/"
        f"{display(worker_max)}, cgroup(current/limit/peak)="
        f"{display(cgroup_current)}/{display(cgroup_limit)}/{display(cgroup_peak)}, "
        f"cgroup_oom({cgroup_oom_status()})"
    )


def completed_parts(task_dir: Path, source: str, task_id: int) -> List[int]:
    pattern = re.compile(
        rf"^{re.escape(source)}-t{task_id:03d}-p(\d{{5}})_text_document\.done$"
    )
    ids = []
    for path in task_dir.glob("*.done"):
        match = pattern.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    ids.sort()
    if ids and ids != list(range(ids[-1] + 1)):
        raise RuntimeError(f"non-contiguous completed parts in {task_dir}: {ids}")
    return ids


def part_stem(source: str, task_id: int, part_id: int) -> str:
    return f"{source}-t{task_id:03d}-p{part_id:05d}_text_document"


def completed_part_totals(
    task_dir: Path, source: str, task_id: int, part_ids: Sequence[int]
) -> Dict[str, int]:
    totals = {"lines": 0, "documents": 0, "tokens": 0, "bad_records": 0}
    for part_id in part_ids:
        meta = load_json(task_dir / f"{part_stem(source, task_id, part_id)}.meta.json")
        totals["lines"] += int(meta["raw_lines"])
        totals["documents"] += int(meta["documents"])
        totals["tokens"] += int(meta["tokens"])
        totals["bad_records"] += int(meta["bad_records"])
    return totals


def build_progress_payload(
    *,
    source: str,
    task_id: int,
    num_tasks: int,
    status: str,
    completed: bool,
    completed_parts_before_run: int,
    completed_parts: int,
    files_total: int,
    cursor: Optional[Cursor],
    committed_before_run: Dict[str, int],
    run: Dict[str, int],
    elapsed_seconds: float,
    inflight_batches: int,
    memory_status: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    run_documents = int(run["documents"])
    run_tokens = int(run["tokens"])
    cumulative = {
        key: int(committed_before_run[key]) + int(run[key])
        for key in ("lines", "documents", "tokens", "bad_records")
    }
    payload: Dict[str, Any] = {
        "updated_at_utc": utc_now(),
        "status": status,
        "completed": completed,
        "source": source,
        "task_id": task_id,
        "num_tasks": num_tasks,
        "completed_parts_before_run": completed_parts_before_run,
        "completed_parts": completed_parts,
        "files_total": files_total,
        "current_file_number": cursor.file_index + 1 if cursor else None,
        "current_cursor": cursor.as_dict() if cursor else None,
        "committed_before_run": dict(committed_before_run),
        "run": {
            **run,
            "elapsed_seconds": elapsed_seconds,
            "docs_per_second": run_documents / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0,
            "mtok_per_second": run_tokens / elapsed_seconds / 1e6
            if elapsed_seconds > 0
            else 0.0,
        },
        "cumulative": cumulative,
        "inflight_batches": inflight_batches,
        "memory": memory_status,
    }
    if error is not None:
        payload["error"] = error
    return payload


def cleanup_uncommitted(task_dir: Path, source: str, task_id: int, completed: Sequence[int]) -> None:
    completed_set = set(completed)
    pattern = re.compile(
        rf"^{re.escape(source)}-t{task_id:03d}-p(\d{{5}})_text_document"
    )
    for path in task_dir.iterdir():
        match = pattern.match(path.name)
        if match and int(match.group(1)) not in completed_set:
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)


def read_resume_cursor(
    task_dir: Path, source: str, task_id: int, completed: Sequence[int]
) -> Optional[Cursor]:
    if not completed:
        return None
    stem = part_stem(source, task_id, completed[-1])
    meta = load_json(task_dir / f"{stem}.meta.json")
    raw = meta["last_input_cursor"]
    return Cursor(
        file_index=int(raw["file_index"]),
        relative_path=str(raw["relative_path"]),
        line_number=int(raw["line_number"]),
    )


def iter_raw_records(
    files: Sequence[InputFile],
    resume_cursor: Optional[Cursor],
    resume_log_interval: int,
) -> Iterator[Tuple[int, str, int, bytes]]:
    for item in files:
        if resume_cursor is not None and item.index < resume_cursor.file_index:
            continue
        if (
            resume_cursor is not None
            and item.index == resume_cursor.file_index
            and item.relative_path != resume_cursor.relative_path
        ):
            raise RuntimeError(
                "resume cursor path does not match current manifest: "
                f"cursor={resume_cursor.relative_path}, manifest={item.relative_path}"
            )

        try:
            with open_compressed_binary(item.path) as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if (
                        resume_cursor is not None
                        and item.index == resume_cursor.file_index
                        and line_number <= resume_cursor.line_number
                    ):
                        if line_number % resume_log_interval == 0:
                            print(
                                f"[resume-scan] {item.relative_path}: skipped "
                                f"{line_number:,}/{resume_cursor.line_number:,} "
                                "already-committed lines",
                                file=sys.stderr,
                            )
                        if line_number == resume_cursor.line_number:
                            print(
                                f"[resume-scan] reached committed cursor "
                                f"{item.relative_path}:{line_number:,}; tokenization resumes next",
                                file=sys.stderr,
                            )
                        continue
                    yield item.index, item.relative_path, line_number, raw_line
        except Exception as exc:
            raise RuntimeError(
                f"failed while reading compressed input {item.path}: {limited_repr(exc)}"
            ) from exc


def record_bad_detail(task_dir: Path, source: str, task_id: int, result: EncodeResult) -> None:
    path = task_dir / f"{source}-t{task_id:03d}.bad-records.jsonl"
    row = {
        "cursor": result.cursor.as_dict(),
        "error_kind": result.error_kind,
        "error_message": result.error_message,
        "raw_bytes": result.raw_bytes,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def refresh_prefix_list(task_dir: Path, source: str, task_id: int) -> None:
    ids = completed_parts(task_dir, source, task_id)
    prefixes = [str((task_dir / part_stem(source, task_id, part_id)).resolve()) for part_id in ids]
    temp = task_dir / "indexed_prefixes.txt.tmp"
    final = task_dir / "indexed_prefixes.txt"
    write_text_atomic(temp, final, "".join(prefix + "\n" for prefix in prefixes))


def process_task(
    *,
    args: argparse.Namespace,
    source: str,
    data_dir: Path,
    output_dir: Path,
    megatron_root: Path,
) -> None:
    all_paths = discover_source_files(data_dir, source)
    if not all_paths:
        print(f"[{source}] no matching files found; skipping", file=sys.stderr)
        return

    files, task_totals = assign_files_by_size(
        all_paths, data_dir, args.num_tasks, args.task_id
    )
    if not files:
        print(
            f"[{source}] task {args.task_id}/{args.num_tasks} received no files; skipping",
            file=sys.stderr,
        )
        return

    task_dir = (
        output_dir
        / source
        / f"task-{args.task_id:03d}-of-{args.num_tasks:03d}"
    )
    if args.reset and task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    lock_path = task_dir / ".preprocess.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another process is already writing {task_dir}") from exc

    run_metadata_path = task_dir / "run_metadata.json"
    task_done_path = task_dir / "task.done.json"
    progress_path = task_dir / "progress.json"
    progress_tmp_path = task_dir / "progress.json.tmp"
    shared_auto_mode = getattr(args, "shared_auto_tokenizer_mode", None)
    if args.use_fast_tokenizer is None and shared_auto_mode is not None:
        use_fast_tokenizer = shared_auto_mode
        tokenizer_mode_source = "existing output shared across --source all"
    else:
        use_fast_tokenizer, tokenizer_mode_source = resolve_tokenizer_mode(
            args.use_fast_tokenizer, run_metadata_path
        )
    print(
        f"[{source}] tokenizer mode={'fast' if use_fast_tokenizer else 'slow'} "
        f"({tokenizer_mode_source})",
        file=sys.stderr,
    )

    manifest_rows, manifest_sha = manifest_payload(files)

    insert_megatron_root(megatron_root)
    # Importing any megatron.core submodule currently initializes optional CUDA
    # inference facilities. Those warnings are unrelated to this CPU pipeline;
    # workers no longer import Megatron at all.
    logging.getLogger("megatron.core.utils").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Transformer Engine and Apex are not installed.*"
        )
        warnings.filterwarnings(
            "ignore", message="Failed to create unified memory mempool.*"
        )
        from megatron.core.datasets import indexed_dataset
        from megatron.core.tokenizers import MegatronTokenizer

    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=str(args.tokenizer_model),
        metadata_path={"library": "huggingface"},
        use_fast=use_fast_tokenizer,
    )
    vocab_size = int(tokenizer.vocab_size)
    eod_id = tokenizer.eod
    if eod_id is None:
        raise RuntimeError("OLMo tokenizer has no EOS/EOD token")
    eod_id = int(eod_id)
    if not 0 <= eod_id < vocab_size:
        raise RuntimeError(f"invalid eod_id={eod_id} for vocab_size={vocab_size}")

    probe = tokenizer.tokenize("Megatron tokenizer verification.\n\n")
    if not probe or any(not isinstance(token_id, int) for token_id in probe):
        raise RuntimeError("Megatron tokenizer self-test failed")

    dtype = indexed_dataset.DType.optimal_dtype(vocab_size)
    dtype_itemsize = int(indexed_dataset.DType.size(dtype))

    tokenizer_meta = tokenizer_fingerprint(args.tokenizer_model)
    tokenizer_meta.update(
        {
            "type": "HuggingFaceTokenizer",
            "megatron_wrapper": type(tokenizer).__name__,
            "vocab_size": vocab_size,
            "eod_id": eod_id,
            "dtype": dtype.__name__,
            "use_fast_tokenizer": use_fast_tokenizer,
            "include_special_tokens": False,
        }
    )
    # Workers construct their own tokenizers. Do not retain an otherwise unused
    # copy in the parent for the full multi-day preprocessing run.
    del tokenizer
    gc.collect()

    max_inflight_batches = args.max_inflight_batches or (2 * args.workers)
    megatron_meta = git_metadata(megatron_root)
    provenance = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "megatron": megatron_meta,
        "data_dir": str(data_dir.resolve()),
        "tokenizer_path": str(args.tokenizer_model.resolve()),
    }
    runtime_config = {
        "workers": args.workers,
        "pool_chunksize": args.pool_chunksize,
        "max_inflight_batches": max_inflight_batches,
        "maxtasksperchild": args.maxtasksperchild,
        "start_method": args.start_method,
        "stall_log_interval": args.stall_log_interval,
        "batch_timeout_seconds": args.batch_timeout_seconds,
        "resume_log_interval": args.resume_log_interval,
        "tokenizer_mode_source": tokenizer_mode_source,
        "worker_tokenizer_backend": "transformers.AutoTokenizer",
    }
    semantic_config = {
        "pipeline_semantics_version": PIPELINE_SEMANTICS_VERSION,
        "source": source,
        "task_id": args.task_id,
        "num_tasks": args.num_tasks,
        "input_manifest_sha256": manifest_sha,
        "tokenizer_fingerprint_sha256": tokenizer_meta["fingerprint_sha256"],
        "vocab_size": vocab_size,
        "eod_id": eod_id,
        "dtype": dtype.__name__,
        "use_fast_tokenizer": use_fast_tokenizer,
        "max_bin_bytes": args.max_bin_bytes,
        "max_documents": args.max_documents,
        "sentence_splitting": False,
        "append_eod": True,
        "error_policy": args.error_policy,
    }
    semantic_config = normalize_semantic_config(semantic_config)
    semantic_config_sha = config_digest(semantic_config)

    original_stable_config_sha = semantic_config_sha
    if run_metadata_path.exists():
        existing = load_json(run_metadata_path)
        original_stable_config_sha = existing.get(
            "stable_config_sha256", semantic_config_sha
        )
        existing_semantic_config = semantic_config_from_metadata(existing)
        mismatches = semantic_config_mismatches(
            existing_semantic_config, semantic_config
        )
        if mismatches:
            details = "\n".join(
                f"  - {field}: existing={old!r}, current={new!r}"
                for field, old, new in mismatches
            )
            fast_mismatch = next(
                (
                    (old, new)
                    for field, old, new in mismatches
                    if field == "use_fast_tokenizer"
                ),
                None,
            )
            fast_hint = ""
            if fast_mismatch == (False, True):
                fast_hint = (
                    "\nA legacy run without use_fast_tokenizer is treated as slow; "
                    "pass --slow-tokenizer to continue it."
                )
            elif fast_mismatch is not None:
                fast_hint = (
                    "\nUse the same --use-fast-tokenizer/--slow-tokenizer mode as "
                    "the original run."
                )
            raise RuntimeError(
                f"existing output has resume-incompatible token/data semantics in "
                f"{task_dir}:\n{details}{fast_hint}\n"
                "Use matching arguments, or use a new --output-dir. Only use --reset "
                "if you intentionally want to discard completed parts."
            )
        append_jsonl(
            task_dir / "resume_history.jsonl",
            {
                "validated_at_utc": utc_now(),
                "semantic_config_sha256": semantic_config_sha,
                "original_stable_config_sha256": original_stable_config_sha,
                "runtime": runtime_config,
                "provenance": provenance,
            },
        )
        print(
            f"[{source}] existing output is resume-compatible; provenance/runtime "
            "changes are recorded in resume_history.jsonl",
            file=sys.stderr,
        )
    else:
        run_metadata = {
            "created_at_utc": utc_now(),
            "semantic_config_sha256": semantic_config_sha,
            "semantic_config": semantic_config,
            # Keep these aliases for readers created for earlier metadata files.
            "stable_config_sha256": semantic_config_sha,
            "stable_config": semantic_config,
            "runtime": runtime_config,
            "provenance": provenance,
            "megatron": megatron_meta,
            "tokenizer": tokenizer_meta,
            "discovered_source_files": len(all_paths),
            "assigned_files": len(files),
            "assigned_compressed_bytes": sum(item.size_bytes for item in files),
            "all_task_compressed_bytes": task_totals,
        }
        write_json_atomic(
            task_dir / "run_metadata.json.tmp", run_metadata_path, run_metadata
        )

    # Only publish the current manifest after resume compatibility succeeds, so
    # a mismatched invocation cannot overwrite the original run's audit record.
    write_manifest(task_dir / "input_manifest.jsonl", manifest_rows)

    if task_done_path.exists():
        task_done = load_json(task_done_path)
        done_parts = completed_parts(task_dir, source, args.task_id)
        done_part_totals = completed_part_totals(
            task_dir, source, args.task_id, done_parts
        )
        trailing_bad = int(task_done.get("trailing_bad_records_without_output_part", 0))
        final_bad_records = int(
            task_done.get(
                "bad_records_total",
                int(task_done.get("bad_records_in_parts", done_part_totals["bad_records"]))
                + trailing_bad,
            )
        )
        final_totals = {
            "lines": int(
                task_done.get("raw_lines", done_part_totals["lines"] + trailing_bad)
            ),
            "documents": int(task_done["documents"]),
            "tokens": int(task_done["tokens"]),
            "bad_records": final_bad_records,
        }
        last_cursor_raw = task_done.get("last_input_cursor")
        last_cursor = (
            Cursor(
                file_index=int(last_cursor_raw["file_index"]),
                relative_path=str(last_cursor_raw["relative_path"]),
                line_number=int(last_cursor_raw["line_number"]),
            )
            if last_cursor_raw
            else None
        )
        write_json_atomic(
            progress_tmp_path,
            progress_path,
            build_progress_payload(
                source=source,
                task_id=args.task_id,
                num_tasks=args.num_tasks,
                status="completed",
                completed=True,
                completed_parts_before_run=len(done_parts),
                completed_parts=len(done_parts),
                files_total=len(files),
                cursor=last_cursor,
                committed_before_run=final_totals,
                run={"lines": 0, "documents": 0, "tokens": 0, "bad_records": 0},
                elapsed_seconds=0.0,
                inflight_batches=0,
                memory_status=None,
            ),
        )
        print(
            f"[{source}] task already complete: parts={len(done_parts)}, "
            f"docs={final_totals['documents']:,}, tokens={final_totals['tokens']:,}, "
            f"progress={progress_path}",
            file=sys.stderr,
        )
        refresh_prefix_list(task_dir, source, args.task_id)
        return

    completed = completed_parts(task_dir, source, args.task_id)
    cleanup_uncommitted(task_dir, source, args.task_id, completed)
    resume_cursor = read_resume_cursor(task_dir, source, args.task_id, completed)
    next_part_id = completed[-1] + 1 if completed else 0
    committed_before_run = completed_part_totals(
        task_dir, source, args.task_id, completed
    )
    refresh_prefix_list(task_dir, source, args.task_id)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source,
                    "task_id": args.task_id,
                    "num_tasks": args.num_tasks,
                    "files": len(files),
                    "compressed_bytes": sum(item.size_bytes for item in files),
                    "manifest_sha256": manifest_sha,
                    "output_dir": str(task_dir),
                    "vocab_size": vocab_size,
                    "eod_id": eod_id,
                    "dtype": dtype.__name__,
                    "use_fast_tokenizer": use_fast_tokenizer,
                },
                indent=2,
            )
        )
        return

    print(
        f"[{source}] files={len(files)}, compressed={human_bytes(sum(f.size_bytes for f in files))}, "
        f"workers={args.workers}, completed_parts={len(completed)}, "
        f"committed_lines={committed_before_run['lines']:,}, "
        f"committed_docs={committed_before_run['documents']:,}, "
        f"committed_tokens={committed_before_run['tokens']:,}, "
        f"committed_bad={committed_before_run['bad_records']:,}, "
        f"resume={resume_cursor.as_dict() if resume_cursor else None}, "
        f"progress={progress_path}",
        file=sys.stderr,
    )
    print(
        f"[{source}] tokenizer vocab={vocab_size}, eod={eod_id}, dtype={dtype.__name__}, "
        f"fast={use_fast_tokenizer}, "
        f"roll={human_bytes(args.max_bin_bytes)} or {args.max_documents:,} documents",
        file=sys.stderr,
    )
    print(
        f"[{source}] bounded scheduling: batch_size={args.pool_chunksize}, "
        f"max_inflight_batches={max_inflight_batches}, "
        f"max_queued_records={args.pool_chunksize * max_inflight_batches}, "
        f"maxtasksperchild={args.maxtasksperchild or 0}",
        file=sys.stderr,
    )

    current: Optional[PartWriter] = None
    pending = PendingPartStats()
    run_started = time.monotonic()
    processed_lines = 0
    processed_docs = 0
    processed_tokens = 0
    bad_records = 0
    final_cursor = resume_cursor

    def run_totals() -> Dict[str, int]:
        return {
            "lines": processed_lines,
            "documents": processed_docs,
            "tokens": processed_tokens,
            "bad_records": bad_records,
        }

    def write_progress(
        status: str,
        *,
        inflight_batches: int = 0,
        memory_status: Optional[str] = None,
        completed_task: bool = False,
        error: Optional[str] = None,
    ) -> None:
        write_json_atomic(
            progress_tmp_path,
            progress_path,
            build_progress_payload(
                source=source,
                task_id=args.task_id,
                num_tasks=args.num_tasks,
                status=status,
                completed=completed_task,
                completed_parts_before_run=len(completed),
                completed_parts=next_part_id,
                files_total=len(files),
                cursor=final_cursor,
                committed_before_run=committed_before_run,
                run=run_totals(),
                elapsed_seconds=max(time.monotonic() - run_started, 0.0),
                inflight_batches=inflight_batches,
                memory_status=memory_status,
                error=error,
            ),
        )

    write_progress("starting")

    def new_writer(part_id: int) -> PartWriter:
        return PartWriter(
            indexed_dataset_module=indexed_dataset,
            output_dir=task_dir,
            source=source,
            task_id=args.task_id,
            part_id=part_id,
            dtype=dtype,
            dtype_itemsize=dtype_itemsize,
            eod_id=eod_id,
            manifest_sha256=manifest_sha,
            tokenizer_metadata=tokenizer_meta,
            run_config=semantic_config,
        )

    ctx = mp.get_context(args.start_method)
    raw_iterator = iter_raw_records(
        files, resume_cursor, resume_log_interval=args.resume_log_interval
    )
    with ctx.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
        initargs=(
            str(args.tokenizer_model),
            use_fast_tokenizer,
            vocab_size,
            eod_id,
        ),
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        def report_wait(inflight_batches: int, wait_seconds: float) -> None:
            memory_status = pool_memory_status(pool)
            write_progress(
                "waiting",
                inflight_batches=inflight_batches,
                memory_status=memory_status,
            )
            print(
                f"[{source}] still waiting for oldest ordered tokenization batch "
                f"({wait_seconds:.0f}s); inflight_batches={inflight_batches}; "
                f"last_input={final_cursor.as_dict() if final_cursor else None}; "
                f"{memory_status}",
                file=sys.stderr,
            )

        try:
            results = iter_bounded_ordered_results(
                pool,
                raw_iterator,
                batch_size=args.pool_chunksize,
                max_inflight_batches=max_inflight_batches,
                heartbeat_seconds=args.stall_log_interval,
                batch_timeout_seconds=args.batch_timeout_seconds,
                on_wait=report_wait,
            )
            for result, inflight_batches in results:
                processed_lines += 1
                final_cursor = result.cursor

                if result.error_kind == "tokenizer_error":
                    raise RuntimeError(
                        f"tokenizer failed at {result.cursor.relative_path}:"
                        f"{result.cursor.line_number}: {result.error_message}"
                    )

                if not result.ok:
                    bad_records += 1
                    record_bad_detail(task_dir, source, args.task_id, result)
                    if args.error_policy == "fail":
                        raise RuntimeError(
                            f"bad record at {result.cursor.relative_path}:"
                            f"{result.cursor.line_number}: {result.error_kind} "
                            f"{result.error_message or ''}"
                        )
                    if current is None:
                        pending.add(result)
                    else:
                        current.observe_line(result)
                else:
                    if current is None:
                        current = new_writer(next_part_id)
                        pending.transfer_to(current)
                    current.observe_line(result)
                    current.add_document(result)
                    processed_docs += 1
                    assert result.token_ids is not None
                    processed_tokens += len(result.token_ids)

                    if (
                        current.bin_bytes >= args.max_bin_bytes
                        or current.documents >= args.max_documents
                    ):
                        prefix = current.finalize()
                        print(
                            f"[{source}] finalized {prefix.name}: docs={current.documents:,}, "
                            f"tokens={current.tokens:,}, bin={human_bytes(current.bin_bytes)}, "
                            f"bad={current.bad_records:,}",
                            file=sys.stderr,
                        )
                        next_part_id += 1
                        current = None
                        refresh_prefix_list(task_dir, source, args.task_id)

                if processed_lines % args.log_interval == 0:
                    elapsed = max(time.monotonic() - run_started, 1e-9)
                    cumulative_lines = committed_before_run["lines"] + processed_lines
                    cumulative_docs = committed_before_run["documents"] + processed_docs
                    cumulative_tokens = committed_before_run["tokens"] + processed_tokens
                    cumulative_bad = committed_before_run["bad_records"] + bad_records
                    memory_status = pool_memory_status(pool)
                    write_progress(
                        "running",
                        inflight_batches=inflight_batches,
                        memory_status=memory_status,
                    )
                    print(
                        f"[{source}] run_lines={processed_lines:,}, "
                        f"cumulative_lines={cumulative_lines:,}, "
                        f"file={result.cursor.file_index + 1:,}/{len(files):,}, "
                        f"input={result.cursor.relative_path}:{result.cursor.line_number:,}, "
                        f"run_docs={processed_docs:,}, cumulative_docs={cumulative_docs:,}, "
                        f"run_tokens={processed_tokens:,}, cumulative_tokens={cumulative_tokens:,}, "
                        f"run_bad={bad_records:,}, cumulative_bad={cumulative_bad:,}, "
                        f"docs/s={processed_docs / elapsed:,.1f}, "
                        f"Mtok/s={processed_tokens / elapsed / 1e6:,.3f}, "
                        f"inflight_batches={inflight_batches}; {memory_status}",
                        file=sys.stderr,
                    )
        except (BrokenPipeError, EOFError, ConnectionError) as exc:
            memory_status = pool_memory_status(pool)
            with contextlib.suppress(Exception):
                write_progress(
                    "failed",
                    memory_status=memory_status,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise RuntimeError(
                "tokenizer worker IPC failed. BrokenPipe is commonly secondary to the "
                "main process or workers being killed by the cgroup OOM killer. Check "
                "the last cgroup/RSS progress line and kernel logs; retry with fewer "
                f"--workers and a smaller --pool-chunksize. Last memory: "
                f"{memory_status}"
            ) from exc
        except WorkerBatchTimeoutError as exc:
            memory_status = pool_memory_status(pool)
            with contextlib.suppress(Exception):
                write_progress(
                    "failed",
                    memory_status=memory_status,
                    error=str(exc),
                )
            raise RuntimeError(
                f"{exc}. This avoids waiting forever after an OOM-killed worker. "
                f"Last memory: {memory_status}"
            ) from exc
        except Exception as exc:
            memory_status = pool_memory_status(pool)
            with contextlib.suppress(Exception):
                write_progress(
                    "failed",
                    memory_status=memory_status,
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    trailing_bad_records = 0
    if current is not None:
        prefix = current.finalize()
        print(
            f"[{source}] finalized {prefix.name}: docs={current.documents:,}, "
            f"tokens={current.tokens:,}, bin={human_bytes(current.bin_bytes)}, "
            f"bad={current.bad_records:,}",
            file=sys.stderr,
        )
        next_part_id += 1
        refresh_prefix_list(task_dir, source, args.task_id)
    elif pending.results:
        # These are trailing rejected lines after the last committed part. They do not
        # justify an empty .bin/.idx, but are represented in task.done.json.
        trailing_bad_records = len(pending.results)
        pending.clear()

    cumulative_parts = completed_parts(task_dir, source, args.task_id)
    cumulative_part_totals = completed_part_totals(
        task_dir, source, args.task_id, cumulative_parts
    )
    cumulative_docs = cumulative_part_totals["documents"]
    cumulative_tokens = cumulative_part_totals["tokens"]
    cumulative_bad = cumulative_part_totals["bad_records"]
    cumulative_lines = committed_before_run["lines"] + processed_lines

    task_done = {
        "completed_at_utc": utc_now(),
        "source": source,
        "task_id": args.task_id,
        "num_tasks": args.num_tasks,
        "parts": len(cumulative_parts),
        "raw_lines": cumulative_lines,
        "documents": cumulative_docs,
        "tokens": cumulative_tokens,
        "bad_records_in_parts": cumulative_bad,
        "trailing_bad_records_without_output_part": trailing_bad_records,
        "bad_records_total": cumulative_bad + trailing_bad_records,
        "last_input_cursor": final_cursor.as_dict() if final_cursor else None,
        "input_manifest_sha256": manifest_sha,
        "semantic_config_sha256": semantic_config_sha,
        "stable_config_sha256": original_stable_config_sha,
    }
    write_json_atomic(task_dir / "task.done.json.tmp", task_done_path, task_done)
    write_progress("completed", completed_task=True)
    elapsed = max(time.monotonic() - run_started, 1e-9)
    print(
        f"[{source}] task complete: parts={len(cumulative_parts)}, docs={cumulative_docs:,}, "
        f"tokens={cumulative_tokens:,}, elapsed={elapsed / 3600:.2f}h",
        file=sys.stderr,
    )


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a locally sampled OLMo-Mix data/ tree to Megatron IndexedDataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="The outer OLMo data directory containing dclm/, pes2o/, starcoder/, wiki/.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help="Local directory containing the downloaded OLMo-2-1124-7B tokenizer/model files.",
    )
    tokenizer_group = parser.add_mutually_exclusive_group()
    tokenizer_group.add_argument(
        "--use-fast-tokenizer",
        dest="use_fast_tokenizer",
        action="store_true",
        help=(
            "Force Hugging Face's fast tokenizer. Without either tokenizer flag, "
            "resume uses the original mode and new output uses fast."
        ),
    )
    tokenizer_group.add_argument(
        "--slow-tokenizer",
        dest="use_fast_tokenizer",
        action="store_false",
        help=(
            "Force the slow Python tokenizer. Without either tokenizer flag, "
            "resume uses the original mode and new output uses fast."
        ),
    )
    parser.set_defaults(use_fast_tokenizer=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root. Default: <data-dir parent>/megatron_indexed",
    )
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=None,
        help="Megatron-LM repository root. Usually auto-detected when run inside the repository.",
    )
    parser.add_argument(
        "--source",
        choices=("all",) + SOURCE_ORDER,
        default="all",
        help="Process one source or all four sources sequentially.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Tokenizer worker processes for this invocation. Default: 24.",
    )
    parser.add_argument(
        "--max-bin-gib",
        type=float,
        default=16.0,
        help="Roll after the .bin reaches this many GiB, at a document boundary. Default: 16.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=10_000_000,
        help="Secondary per-part document limit. Default: 10,000,000.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Deterministically split this source's files into N size-balanced independent tasks.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Zero-based task id when --num-tasks > 1.",
    )
    parser.add_argument(
        "--pool-chunksize",
        type=int,
        default=16,
        help="Records per explicitly bounded worker batch. Default: 16.",
    )
    parser.add_argument(
        "--max-inflight-batches",
        type=int,
        default=0,
        help=(
            "Maximum queued/running/result batches, which bounds parent-side "
            "out-of-order memory. 0 selects 2 * workers. Default: 0."
        ),
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=10_000,
        help=(
            "Recycle tokenizer workers after this many batches; 0 disables "
            "recycling. Default: 10,000."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10_000,
        help="Print progress every N input lines. Default: 10,000.",
    )
    parser.add_argument(
        "--resume-log-interval",
        type=int,
        default=100_000,
        help=(
            "Print progress while rescanning a compressed shard to its committed "
            "resume cursor. Default: 100,000 lines."
        ),
    )
    parser.add_argument(
        "--stall-log-interval",
        type=float,
        default=60.0,
        help=(
            "Print a memory heartbeat while waiting for an ordered batch. "
            "Default: 60 seconds."
        ),
    )
    parser.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=1800.0,
        help=(
            "Abort if the oldest batch produces no result for this long, which "
            "prevents an OOM-killed worker from hanging forever. 0 disables the "
            "timeout. Default: 1800 seconds."
        ),
    )
    parser.add_argument(
        "--error-policy",
        choices=("skip", "fail"),
        default="skip",
        help="Skip malformed/empty records with logs, or fail immediately. Compression and tokenizer errors always fail.",
    )
    parser.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method. Default: spawn.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files, validate tokenizer/config, and write manifests without tokenizing.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this source/task output directory before starting. Use carefully.",
    )
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.tokenizer_model = args.tokenizer_model.resolve()
    if args.output_dir is None:
        args.output_dir = (args.data_dir.parent / "megatron_indexed").resolve()
    else:
        args.output_dir = args.output_dir.resolve()

    if not args.data_dir.is_dir():
        parser.error(f"--data-dir is not a directory: {args.data_dir}")
    if not args.tokenizer_model.exists():
        parser.error(f"--tokenizer-model does not exist: {args.tokenizer_model}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_bin_gib <= 0:
        parser.error("--max-bin-gib must be > 0")
    if args.max_documents < 1:
        parser.error("--max-documents must be >= 1")
    if args.pool_chunksize < 1:
        parser.error("--pool-chunksize must be >= 1")
    if args.max_inflight_batches < 0:
        parser.error("--max-inflight-batches must be >= 0")
    if args.log_interval < 1:
        parser.error("--log-interval must be >= 1")
    if args.resume_log_interval < 1:
        parser.error("--resume-log-interval must be >= 1")
    if args.stall_log_interval <= 0:
        parser.error("--stall-log-interval must be > 0")
    if args.batch_timeout_seconds < 0:
        parser.error("--batch-timeout-seconds must be >= 0")
    if args.maxtasksperchild == 0:
        args.maxtasksperchild = None
    elif args.maxtasksperchild < 0:
        parser.error("--maxtasksperchild must be >= 0")
    if args.source == "all" and (args.num_tasks != 1 or args.task_id != 0):
        parser.error("--source all currently requires --num-tasks 1 --task-id 0")

    args.max_bin_bytes = int(args.max_bin_gib * 1024**3)
    return args


def main() -> None:
    args = parse_args()
    configure_process_environment()
    megatron_root = find_megatron_root(args.megatron_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "all" and args.use_fast_tokenizer is None:
        args.shared_auto_tokenizer_mode = shared_tokenizer_mode_for_all_sources(
            args.output_dir, args.task_id, args.num_tasks
        )

    sources = SOURCE_ORDER if args.source == "all" else (args.source,)
    for source in sources:
        process_task(
            args=args,
            source=source,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            megatron_root=megatron_root,
        )


if __name__ == "__main__":
    main()
