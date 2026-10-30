#!/usr/bin/env python3

import argparse
import gzip
import io
import json
from contextlib import contextmanager
from pathlib import Path

import zstandard as zstd


@contextmanager
def open_text(path: Path):
    name = path.name.lower()

    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream
        return

    if name.endswith((".zstd", ".zst")):
        with path.open("rb") as raw:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                    yield stream
        return

    with path.open("rt", encoding="utf-8") as stream:
        yield stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    with open_text(args.path) as stream:
        first_line = stream.readline()

    if not first_line:
        raise RuntimeError("File is empty")

    stripped = first_line.lstrip()
    print("first character:", repr(stripped[:1]))
    print("compressed file:", args.path)
    print("first-line bytes:", len(first_line.encode("utf-8")))

    if stripped.startswith("["):
        print("Format appears to be a JSON array, not JSON Lines")
        return

    record = json.loads(first_line)

    if not isinstance(record, dict):
        raise TypeError(f"First record is {type(record).__name__}, expected object")

    text = record.get("text")

    print("keys:", sorted(record.keys()))
    print("field types:", {key: type(value).__name__ for key, value in record.items()})
    print("text type:", type(text).__name__)
    print("text characters:", len(text) if isinstance(text, str) else None)
    print("text preview:", repr(text[:200]) if isinstance(text, str) else None)


if __name__ == "__main__":
    main()
