"""Tests for GGUF patch script (.github/tier2/patch_gguf_add_bos.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


def create_synthetic_gguf(path: Path) -> None:
    """Create a minimal synthetic mistral3 GGUF file with F32 and Q8_0 tensors."""
    writer = gguf.GGUFWriter(str(path), arch="mistral3")

    # Add tokenizer metadata
    tokens = ["<unk>", "<s>", "</s>", "hello", "world"]
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([1, 2, 3, 1, 1])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    # Add F32 tensor
    f32_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    writer.add_tensor("test.weight.f32", f32_data, raw_dtype=gguf.GGMLQuantizationType.F32)

    # Add Q8_0 tensor
    # 32 elements (one block of 32 for Q8_0)
    raw_f32 = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    q8_data = gguf.quants.quantize(raw_f32, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("test.weight.q8_0", q8_data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_patch_gguf_success_and_idempotency(tmp_path: Path) -> None:
    """Test successful patching of GGUF and idempotency on running again."""
    src = tmp_path / "synthetic.gguf"
    dst = tmp_path / "patched.gguf"
    create_synthetic_gguf(src)

    script = Path(".github/tier2/patch_gguf_add_bos.py")

    # Run patch script
    res = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "tensors identical=True" in res.stdout
    assert f"patched {dst}:" in res.stdout
    assert dst.exists()

    # Verify dst contents
    reader = gguf.GGUFReader(str(dst), "r")
    add_bos = reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)
    assert add_bos is not None
    assert add_bos.contents() is False

    # Run again on its own output (idempotency)
    dst2 = tmp_path / "patched_again.gguf"
    res2 = subprocess.run([sys.executable, str(script), str(dst), str(dst2)], capture_output=True, text=True)
    assert res2.returncode == 0, f"Script failed on patched file: {res2.stderr}"
    assert "tensors identical=True" in res2.stdout
    assert dst2.exists()

    reader2 = gguf.GGUFReader(str(dst2), "r")
    add_bos2 = reader2.get_field(gguf.Keys.Tokenizer.ADD_BOS)
    assert add_bos2 is not None
    assert add_bos2.contents() is False


def test_patch_gguf_error_cases(tmp_path: Path) -> None:
    """Test error handling when src does not exist or src == dst."""
    script = Path(".github/tier2/patch_gguf_add_bos.py")

    # Non-existent src
    non_existent = tmp_path / "does_not_exist.gguf"
    dst = tmp_path / "should_not_exist.gguf"
    res = subprocess.run([sys.executable, str(script), str(non_existent), str(dst)], capture_output=True, text=True)
    assert res.returncode != 0
    assert not dst.exists()
    assert "does not exist" in res.stderr

    # src == dst
    src = tmp_path / "synthetic.gguf"
    create_synthetic_gguf(src)
    res_same = subprocess.run([sys.executable, str(script), str(src), str(src)], capture_output=True, text=True)
    assert res_same.returncode != 0
    assert "must be different" in res_same.stderr
