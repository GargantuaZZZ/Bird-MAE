#!/usr/bin/env python3
"""Archive audio files in place while preserving the dataset directory tree."""

from __future__ import annotations

import argparse
import hashlib
import os
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


def discover_audio_by_directory(root: Path) -> list[tuple[Path, list[AudioFile]]]:
    grouped: list[tuple[Path, list[AudioFile]]] = []

    for current_dir, dir_names, file_names in os.walk(root):
        dir_names.sort()
        directory = Path(current_dir)
        audio_files = [
            AudioFile(path=directory / name, size=(directory / name).stat().st_size)
            for name in sorted(file_names)
            if (directory / name).is_file()
            and (directory / name).suffix.lower() in AUDIO_SUFFIXES
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
            raise RuntimeError(
                f"压缩包文件列表校验失败: {archive_path}"
            )

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
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        with tarfile.open(
            temporary_path, mode="w:gz", compresslevel=compression_level
        ) as archive:
            for audio_file in files:
                archive.add(
                    audio_file.path,
                    arcname=audio_file.path.name,
                    recursive=False,
                )

        verify_archive(temporary_path, files)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_directory(
    directory: Path,
    files: list[AudioFile],
    max_size: int,
    max_files: int | None,
    prefix: str,
    compression_level: int,
    dry_run: bool,
) -> tuple[int, int]:
    shards = split_into_shards(files, max_size, max_files)
    output_paths = [
        directory / shard_name(prefix, index, len(shards))
        for index in range(len(shards))
    ]
    conflicts = [path for path in output_paths if path.exists()]
    if conflicts:
        names = ", ".join(path.name for path in conflicts)
        raise FileExistsError(f"{directory} 中已存在目标压缩包: {names}")

    relative_directory = directory.as_posix()
    print(
        f"{relative_directory}: {len(files)} 个音频文件 -> "
        f"{len(shards)} 个压缩包"
    )
    if dry_run:
        for output_path, shard_files in zip(output_paths, shards):
            print(f"  [预览] {output_path.name}: {len(shard_files)} 个文件")
        return len(files), len(shards)

    created_archives: list[Path] = []
    try:
        for output_path, shard_files in zip(output_paths, shards):
            write_verified_archive(output_path, shard_files, compression_level)
            created_archives.append(output_path)
            print(f"  已创建并校验: {output_path.name}")
    except Exception:
        for archive_path in created_archives:
            archive_path.unlink(missing_ok=True)
        raise

    for audio_file in files:
        audio_file.path.unlink()
    print(f"  已删除 {len(files)} 个原始音频文件")
    return len(files), len(shards)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "逐目录压缩直属的 .ogg/.wav 文件；压缩包保存在原目录，"
            "校验成功后删除原音频，其他文件保持不变。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="数据集根目录")
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
        help="只显示将进行的操作，不创建压缩包或删除文件",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files 必须大于 0")
    if not args.prefix or "/" in args.prefix or "\\" in args.prefix:
        raise SystemExit("--prefix 必须是普通文件名，不能包含路径分隔符")

    grouped_files = discover_audio_by_directory(input_dir)
    if not grouped_files:
        raise SystemExit(f"在 {input_dir} 中没有找到 .ogg 或 .wav 文件")

    conflicts = []
    for directory, files in grouped_files:
        shards = split_into_shards(files, args.max_size, args.max_files)
        conflicts.extend(
            directory / shard_name(args.prefix, index, len(shards))
            for index in range(len(shards))
            if (directory / shard_name(args.prefix, index, len(shards))).exists()
        )
    if conflicts:
        conflict_list = "\n".join(f"  - {path}" for path in conflicts)
        raise SystemExit(f"以下目标压缩包已存在，未处理任何文件:\n{conflict_list}")

    total_files = 0
    total_archives = 0
    for directory, files in grouped_files:
        try:
            file_count, archive_count = process_directory(
                directory=directory,
                files=files,
                max_size=args.max_size,
                max_files=args.max_files,
                prefix=args.prefix,
                compression_level=args.compression_level,
                dry_run=args.dry_run,
            )
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            raise SystemExit(f"处理失败: {exc}") from exc
        total_files += file_count
        total_archives += archive_count

    action = "计划处理" if args.dry_run else "处理完成"
    print(
        f"{action}: {len(grouped_files)} 个目录，"
        f"{total_files} 个音频文件，{total_archives} 个压缩包。"
    )


if __name__ == "__main__":
    main()
