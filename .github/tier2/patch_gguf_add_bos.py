#!/usr/bin/env python3
"""Patch GGUF file to write tokenizer.ggml.add_bos_token=false into a derived GGUF.

Diagnostic tool for SkillEvaluator Tier 2 to opt-in fix the implicit BOS offset
in llama.cpp / Ollama for the mistral3 / pixtral embedding models.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def patch_gguf_add_bos(src_path: str, dst_path: str) -> None:
    """Read src_path, patch tokenizer.ggml.add_bos_token=False, and write to dst_path."""
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        print(f"Error: source file '{src}' does not exist.", file=sys.stderr)
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        sys.exit(1)

    try:
        if src.resolve() == dst.resolve():
            print("Error: source and destination files must be different.", file=sys.stderr)
            sys.exit(1)
    except OSError:
        pass

    try:
        import gguf
        from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata
    except ImportError:
        print("Error: 'gguf' package is required. Install with `pip install gguf`.", file=sys.stderr)
        sys.exit(1)

    # Perform patching
    try:
        reader = gguf.GGUFReader(str(src), "r")
        arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
        if arch_field is None:
            print(f"Error: {src} missing general.architecture field.", file=sys.stderr)
            sys.exit(1)
        arch = arch_field.contents()
        writer = gguf.GGUFWriter(str(dst), arch=arch, endianess=reader.endianess)
        al = reader.get_field(gguf.Keys.General.ALIGNMENT)
        if al is not None:
            writer.data_alignment = al.contents()

        copy_with_new_metadata(
            reader,
            writer,
            {gguf.Keys.Tokenizer.ADD_BOS: MetadataDetails(gguf.GGUFValueType.BOOL, False, "disable implicit BOS")},
            [],
        )
    except Exception as e:
        print(f"Error during GGUF patching: {e}", file=sys.stderr)
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        sys.exit(1)

    # Verification: re-open dst and compare against src
    try:
        src_reader = gguf.GGUFReader(str(src), "r")
        dst_reader = gguf.GGUFReader(str(dst), "r")

        # 1. Verify add_bos_token is False
        add_bos_field = dst_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)
        if add_bos_field is None or add_bos_field.contents() is not False:
            raise ValueError(f"Expected {gguf.Keys.Tokenizer.ADD_BOS} to be False, got {add_bos_field}")

        # 2. Check metadata keys added / changed
        # Note: GGUF.kv_count is a file-header field updated when metadata keys are added.
        ignored_header_fields = {"GGUF.kv_count", "GGUF.tensor_count", "GGUF.version"}

        src_fields = {f.name: f for f in src_reader.fields.values()}
        dst_fields = {f.name: f for f in dst_reader.fields.values()}

        added_keys = []
        changed_keys = []

        for name, d_f in dst_fields.items():
            if name in ignored_header_fields:
                continue
            if name not in src_fields:
                added_keys.append(name)
            else:
                s_f = src_fields[name]
                if name == gguf.Keys.Tokenizer.ADD_BOS:
                    if s_f.contents() != d_f.contents():
                        changed_keys.append(name)
                else:
                    # Compare other metadata
                    if s_f.contents() != d_f.contents():
                        raise ValueError(f"Metadata field '{name}' differs between src and dst")

        for name in src_fields:
            if name in ignored_header_fields:
                continue
            if name not in dst_fields:
                raise ValueError(f"Metadata field '{name}' was dropped in dst")

        # 3. Check tensors: same name, type, shape and sha256 of data
        if len(src_reader.tensors) != len(dst_reader.tensors):
            raise ValueError(
                f"Tensor count mismatch: src has {len(src_reader.tensors)}, dst has {len(dst_reader.tensors)}"
            )

        for s_t, d_t in zip(src_reader.tensors, dst_reader.tensors):
            if s_t.name != d_t.name:
                raise ValueError(f"Tensor name mismatch: {s_t.name} vs {d_t.name}")
            if s_t.tensor_type != d_t.tensor_type:
                raise ValueError(f"Tensor {s_t.name} type mismatch: {s_t.tensor_type} vs {d_t.tensor_type}")
            if list(s_t.shape) != list(d_t.shape):
                raise ValueError(f"Tensor {s_t.name} shape mismatch: {s_t.shape} vs {d_t.shape}")
            s_data_hash = hashlib.sha256(s_t.data).hexdigest()
            d_data_hash = hashlib.sha256(d_t.data).hexdigest()
            if s_data_hash != d_data_hash:
                raise ValueError(f"Tensor {s_t.name} data sha256 mismatch: {s_data_hash} vs {d_data_hash}")

    except Exception as e:
        print(f"Verification failed on patched GGUF: {e}", file=sys.stderr)
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        sys.exit(1)

    dst_sha256 = compute_file_sha256(dst)
    print(f"patched {dst}: KV added={added_keys} changed={changed_keys} tensors identical=True sha256={dst_sha256}")


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src.gguf> <dst.gguf>", file=sys.stderr)
        sys.exit(1)

    patch_gguf_add_bos(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
