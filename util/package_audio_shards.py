#!/usr/bin/env python3
"""Package .ogg and .wav files into size-limited tar.gz shards."""

from __future__ import annotations

import argparse
import csv
import tarfile
from dataclasses import dataclass
from pathlib import Path


AUDIO_SUFFIXES = {".ogg", ".wav"}


@dataclass(frozen=True)
class AudioFile:
    path: Path
    archive_name: str
    size: int


def parse_size(value: str) -> int:
    """Parse sizes such as 500MB, 1.5GB, or 1000000."""
    normalized = value.strip().upper().replace("IB", "B")
    units = {
        "TB": 1024**4,
        "GB": 1024**3,
        "MB": 1024**2,
        "KB": 1024,
        "B": 1,
    }

    for suffix, multiplier in units.items():
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            break
    else:
        number = normalized
        multiplier = 1

    try:
        size = int(float(number) * multiplier)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无效大小: {value!r}") from exc

    if size <= 0:
        raise argparse.ArgumentTypeError("分片大小必须大于 0")
    return size


def discover_audio_files(input_dir: Path) -> list[AudioFile]:
    files = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            relative_path = path.relative_to(input_dir)
            files.append(
                AudioFile(
                    path=path,
                    archive_name=relative_path.as_posix(),
                    size=path.stat().st_size,
                )
            )
    return sorted(files, key=lambda item: item.archive_name)


def split_into_shards(
    files: list[AudioFile], max_size: int, max_files: int | None
) -> list[list[AudioFile]]:
    shards: list[list[AudioFile]] = []
    current: list[AudioFile] = []
    current_size = 0

    for audio_file in files:
        size_limit_reached = current and current_size + audio_file.size > max_size
        file_limit_reached = (
            current and max_files is not None and len(current) >= max_files
        )
        if size_limit_reached or file_limit_reached:
            shards.append(current)
            current = []
            current_size = 0

        current.append(audio_file)
        current_size += audio_file.size

    if current:
        shards.append(current)
    return shards


def write_shard(
    files: list[AudioFile],
    output_path: Path,
    compression_level: int,
) -> None:
    with tarfile.open(
        output_path, mode="w:gz", compresslevel=compression_level
    ) as archive:
        for audio_file in files:
            archive.add(
                audio_file.path,
                arcname=audio_file.archive_name,
                recursive=False,
            )


def write_manifest(
    output_dir: Path,
    shard_records: list[tuple[str, AudioFile]],
) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["shard", "path", "size_bytes"])
        for shard_name, audio_file in shard_records:
            writer.writerow(
                [shard_name, audio_file.archive_name, audio_file.size]
            )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="递归查找 .ogg/.wav 文件，并分块压缩为 tar.gz。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="音频数据集根目录")
    parser.add_argument("output_dir", type=Path, help="分片输出目录")
    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=parse_size("1GB"),
        help="每个分片包含的原始文件大小上限，例如 500MB、1GB",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="每个分片最多文件数；默认不限制",
    )
    parser.add_argument(
        "--prefix",
        default="audio",
        help="分片文件名前缀",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(1, 10),
        default=6,
        metavar="1-9",
        help="gzip 压缩等级",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files 必须大于 0")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("输出目录不能位于输入目录内部")

    files = discover_audio_files(input_dir)
    if not files:
        raise SystemExit(f"在 {input_dir} 中没有找到 .ogg 或 .wav 文件")

    shards = split_into_shards(files, args.max_size, args.max_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(5, len(str(len(shards))))
    records: list[tuple[str, AudioFile]] = []

    print(f"找到 {len(files)} 个音频文件，将生成 {len(shards)} 个分片。")
    for index, shard_files in enumerate(shards):
        shard_name = f"{args.prefix}-{index:0{width}d}-of-{len(shards):0{width}d}.tar.gz"
        output_path = output_dir / shard_name
        source_size = sum(audio_file.size for audio_file in shard_files)

        print(
            f"[{index + 1}/{len(shards)}] {shard_name}: "
            f"{len(shard_files)} files, {source_size / 1024**2:.2f} MiB"
        )
        write_shard(shard_files, output_path, args.compression_level)
        records.extend((shard_name, audio_file) for audio_file in shard_files)

    manifest_path = write_manifest(output_dir, records)
    print(f"完成。清单: {manifest_path}")


if __name__ == "__main__":
    main()
