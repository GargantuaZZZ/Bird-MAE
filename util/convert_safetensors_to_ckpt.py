#!/usr/bin/env python3
"""Convert a .safetensors checkpoint to a .ckpt/.pt file using torch.save.

This wraps the tensor dict under the "model" key to match the repository's
pretrained-weight loading logic.
"""

import argparse
import os

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert safetensors to torch checkpoint with a 'model' key."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to .safetensors file")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path (.ckpt or .pt)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    state = load_file(args.input)
    torch.save({"model": state}, args.output)
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
