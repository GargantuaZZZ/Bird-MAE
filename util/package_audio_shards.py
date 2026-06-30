#!/usr/bin/env python3
"""Create a dataset copy with per-directory audio archives."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


AUDIO_SUFFIXES = {".ogg", ".wav"}


@dataclass(frozen=True)
class AudioFile:
    path: Path
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


def is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES


def discover_audio_by_directory(root: Path) -> list[tuple[Path, list[AudioFile]]]:
    grouped: list[tuple[Path, list[AudioFile]]] = []

    for current_dir, dir_names, file_names in os.walk(root):
        dir_names.sort()
        directory = Path(current_dir)
        audio_files = [
            AudioFile(path=directory / name, size=(directory / name).stat().st_size)
            for name in sorted(file_names)
            if is_audio(directory / name)
        ]
        if audio_files:
            grouped.append((directory, audio_files))

    return grouped


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


def shard_name(prefix: str, index: int, shard_count: int) -> str:
    if shard_count == 1:
        return f"{prefix}.tar.gz"
    width = max(5, len(str(shard_count)))
    return f"{prefix}-{index:0{width}d}-of-{shard_count:0{width}d}.tar.gz"


def file_sha256(file_object: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := file_object.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive_path: Path, files: list[AudioFile]) -> None:
    expected = {audio_file.path.name: audio_file for audio_file in files}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = {
            member.name: member
            for member in archive.getmembers()
            if member.isfile()
        }
        if set(members) != set(expected):
            raise RuntimeError(f"压缩包文件列表校验失败: {archive_path}")

        for name, audio_file in expected.items():
            member = members[name]
            if member.size != audio_file.size:
                raise RuntimeError(f"压缩包文件大小校验失败: {name}")
            archived_file = archive.extractfile(member)
            if archived_file is None:
                raise RuntimeError(f"无法读取压缩包中的文件: {name}")
            with audio_file.path.open("rb") as source_file:
                if file_sha256(archived_file) != file_sha256(source_file):
                    raise RuntimeError(f"压缩包文件内容校验失败: {name}")


def write_verified_archive(
    output_path: Path,
    files: list[AudioFile],
    compression_level: int,
) -> None:
    with tarfile.open(
        output_path, mode="w:gz", compresslevel=compression_level
    ) as archive:
        for audio_file in files:
            archive.add(
                audio_file.path,
                arcname=audio_file.path.name,
                recursive=False,
            )
    verify_archive(output_path, files)


def copy_non_audio_tree(input_dir: Path, output_dir: Path) -> int:
    copied_files = 0
    for current_dir, dir_names, file_names in os.walk(input_dir):
        dir_names.sort()
        source_dir = Path(current_dir)
        relative_dir = source_dir.relative_to(input_dir)
        destination_dir = output_dir / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)

        for name in sorted(file_names):
            source_path = source_dir / name
            if is_audio(source_path):
                continue
            shutil.copy2(source_path, destination_dir / name)
            copied_files += 1

    return copied_files


def build_copy(
    input_dir: Path,
    output_dir: Path,
    grouped_files: list[tuple[Path, list[AudioFile]]],
    max_size: int,
    max_files: int | None,
    prefix: str,
    compression_level: int,
) -> tuple[int, int, int]:
    copied_files = copy_non_audio_tree(input_dir, output_dir)
    audio_count = 0
    archive_count = 0

    for source_dir, files in grouped_files:
        relative_dir = source_dir.relative_to(input_dir)
        destination_dir = output_dir / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        shards = split_into_shards(files, max_size, max_files)

        print(
            f"{relative_dir or Path('.')}: {len(files)} 个音频文件 -> "
            f"{len(shards)} 个压缩包"
        )
        for index, shard_files in enumerate(shards):
            output_path = destination_dir / shard_name(prefix, index, len(shards))
            if output_path.exists():
                raise FileExistsError(f"目标文件已存在: {output_path}")
            write_verified_archive(output_path, shard_files, compression_level)
            print(f"  已创建并校验: {output_path.name}")

        audio_count += len(files)
        archive_count += len(shards)

    return copied_files, audio_count, archive_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "创建数据集副本，保留原目录层级和非音频文件；在副本的对应目录中，"
            "将直属 .ogg/.wav 替换为 tar.gz 分片。源目录不会被修改。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="原始数据集根目录")
    parser.add_argument("output_dir", type=Path, help="处理后的副本目录，必须不存在")
    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=parse_size("1GB"),
        help="每个分片包含的原始音频大小上限，例如 500MB、1GB",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="每个分片最多音频文件数；默认不限制",
    )
    parser.add_argument(
        "--prefix",
        default="audio",
        help="每个目录中的压缩包文件名前缀",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(1, 10),
        default=6,
        metavar="1-9",
        help="gzip 压缩等级",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示统计信息，不创建副本",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")
    if output_dir.exists():
        raise SystemExit(f"输出目录已存在，请更换路径: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("输出目录不能位于输入目录内部")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files 必须大于 0")
    if not args.prefix or "/" in args.prefix or "\\" in args.prefix:
        raise SystemExit("--prefix 必须是普通文件名，不能包含路径分隔符")

    grouped_files = discover_audio_by_directory(input_dir)
    audio_count = sum(len(files) for _, files in grouped_files)
    if not grouped_files:
        raise SystemExit(f"在 {input_dir} 中没有找到 .ogg 或 .wav 文件")

    if args.dry_run:
        archive_count = sum(
            len(split_into_shards(files, args.max_size, args.max_files))
            for _, files in grouped_files
        )
        print(
            f"计划创建副本 {output_dir}: {len(grouped_files)} 个含音频目录，"
            f"{audio_count} 个音频文件，{archive_count} 个压缩包。"
        )
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
        )
    )
    try:
        copied_files, audio_count, archive_count = build_copy(
            input_dir=input_dir,
            output_dir=staging_dir,
            grouped_files=grouped_files,
            max_size=args.max_size,
            max_files=args.max_files,
            prefix=args.prefix,
            compression_level=args.compression_level,
        )
        staging_dir.replace(output_dir)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise SystemExit(f"创建副本失败，源目录未修改: {exc}") from exc

    print(
        f"处理完成: {output_dir}\n"
        f"复制 {copied_files} 个非音频文件，压缩 {audio_count} 个音频文件，"
        f"生成 {archive_count} 个压缩包。源目录未修改。"
    )


if __name__ == "__main__":
    main()
