"""Tests for SentenceEmbeddingTransformer.

All tests mock the ``SentenceTransformer`` model so that ``sentence-transformers``
is not required at test-time and tests run in milliseconds without GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer


# ── Fixtures ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 5
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _make_mock_model(embedding_dim: int = EMBEDDING_DIM) -> MagicMock:
    """Return a mock ``SentenceTransformer`` instance."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = embedding_dim

    def _encode(sentences, **kwargs):
        n = len(sentences)
        return np.random.default_rng().standard_normal((n, embedding_dim)).astype(np.float32)

    model.encode.side_effect = _encode
    return model


@pytest.fixture
def patch_st(mocker):
    """Patch ``SentenceTransformer`` so it returns a controllable mock."""
    mock_model = _make_mock_model()
    mock_cls = MagicMock(return_value=mock_model)
    mock_module = MagicMock()
    mock_module.SentenceTransformer = mock_cls

    mocker.patch.dict("sys.modules", {"sentence_transformers": mock_module})
    return mock_cls, mock_model


# ── Dependency guard ──────────────────────────────────────────────────────────


def test_import_error_message(mocker):
    """A clear message is raised when sentence-transformers is missing."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=_fake_import)
    t = SentenceEmbeddingTransformer()
    with pytest.raises(ImportError, match="pip install data-science-utils\\[nlp\\]"):
        t.fit(["hello"])


# ── Constructor defaults ─────────────────────────────────────────────────────


def test_default_parameters():
    """Constructor defaults match the agreed-upon values."""
    t = SentenceEmbeddingTransformer()
    assert t.model_name == DEFAULT_MODEL
    assert t.batch_size == 32
    assert t.show_progress_bar is False
    assert t.normalize_embeddings is False
    assert t.device is None
    assert t.precision == "float32"
    assert t.truncate_dim is None
    assert t.prompt_name is None
    assert t.prompt is None


def test_custom_parameters():
    """Parameters set via constructor are stored correctly."""
    t = SentenceEmbeddingTransformer(
        model_name="custom-model",
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        device="cuda",
        precision="int8",
        truncate_dim=128,
        prompt_name="query",
        prompt="Search: ",
    )
    assert t.model_name == "custom-model"
    assert t.batch_size == 64
    assert t.show_progress_bar is True
    assert t.normalize_embeddings is True
    assert t.device == "cuda"
    assert t.precision == "int8"
    assert t.truncate_dim == 128
    assert t.prompt_name == "query"
    assert t.prompt == "Search: "


# ── fit / lazy loading ────────────────────────────────────────────────────────


def test_fit_loads_model(patch_st):
    """Fit instantiates the SentenceTransformer and sets fitted attributes."""
    mock_cls, _ = patch_st
    t = SentenceEmbeddingTransformer()
    result = t.fit(["hello"])
    assert result is t
    mock_cls.assert_called_once()
    assert t.embedding_dimension_ == EMBEDDING_DIM
    assert t.n_features_in_ == 1


def test_fit_reuses_model(patch_st):
    """Calling fit() twice does NOT reload the model."""
    mock_cls, _ = patch_st
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    t.fit(["b"])
    mock_cls.assert_called_once()


def test_fit_reloads_on_model_reload_params_change(patch_st):
    """If model_name, device, or truncate_dim change, the next fit() call reloads the model."""
    mock_cls, _ = patch_st
    t = SentenceEmbeddingTransformer(model_name="model-A", device="cpu", truncate_dim=128)
    t.fit(["hello"])
    assert mock_cls.call_count == 1

    # Same params -> no reload
    t.fit(["world"])
    assert mock_cls.call_count == 1

    # Different model name -> reload
    t.set_params(model_name="model-B")
    t.fit(["test"])
    assert mock_cls.call_count == 2
    mock_cls.assert_called_with("model-B", device="cpu", truncate_dim=128)

    # Different device -> reload
    t.set_params(device="cuda")
    t.fit(["test"])
    assert mock_cls.call_count == 3
    mock_cls.assert_called_with("model-B", device="cuda", truncate_dim=128)

    # Different truncate_dim -> reload
    t.set_params(truncate_dim=64)
    t.fit(["test"])
    assert mock_cls.call_count == 4
    mock_cls.assert_called_with("model-B", device="cuda", truncate_dim=64)


def test_fit_passes_device_and_truncate_dim(patch_st):
    """Device and truncate_dim are forwarded to the SentenceTransformer constructor."""
    mock_cls, _ = patch_st
    t = SentenceEmbeddingTransformer(device="cuda:0", truncate_dim=128)
    t.fit(["x"])
    mock_cls.assert_called_once_with(DEFAULT_MODEL, device="cuda:0", truncate_dim=128)


def test_fit_invalid_precision():
    """Invalid precision parameter raises ValueError."""
    t = SentenceEmbeddingTransformer(precision="invalid_prec")
    with pytest.raises(ValueError, match="Invalid precision 'invalid_prec'"):
        t.fit(["hello"])


# ── transform ─────────────────────────────────────────────────────────────────


def test_transform_before_fit_raises():
    """Transform without fit() raises NotFittedError."""
    t = SentenceEmbeddingTransformer()
    with pytest.raises(NotFittedError):
        t.transform(["hello"])


@pytest.mark.parametrize(
    ("X", "expected_rows"),
    [
        pytest.param(["x", "y", "z"], 3, id="list_of_strings"),
        pytest.param("hello world", 1, id="single_string"),
        pytest.param(pd.Series(["one", "two", "three"]), 3, id="pandas_series"),
        pytest.param(pd.DataFrame({"text": ["a", "b"]}), 2, id="dataframe_single_col"),
        pytest.param(np.array(["hello", "world"]), 2, id="numpy_1d"),
        pytest.param(np.array([["hello"], ["world"]]), 2, id="numpy_2d_single_col"),
    ],
)
def test_transform_input_formats(patch_st, X, expected_rows):
    """Various input formats produce (n_samples, embedding_dim) arrays."""
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    out = t.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (expected_rows, EMBEDDING_DIM)


# ── Multi-column rejection ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(pd.DataFrame({"a": ["x"], "b": ["y"]}), id="multi_col_dataframe"),
        pytest.param(np.array([["a", "b"], ["c", "d"]]), id="multi_col_ndarray"),
    ],
)
def test_transform_rejects_multi_column(patch_st, X):
    """Multi-column inputs raise ValueError."""
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    with pytest.raises(ValueError, match="single text column"):
        t.transform(X)


# ── None / NaN handling ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("X", "expected_texts"),
    [
        pytest.param(
            pd.Series(["hello", None, np.nan, pd.NA, "world"]),
            ["hello", "", "", "", "world"],
            id="series_none_and_nan_and_pdna",
        ),
        pytest.param(["text", None], ["text", ""], id="list_with_none"),
    ],
)
def test_none_nan_replaced_with_empty(patch_st, X, expected_texts):
    """None and NaN values are replaced with empty strings before encoding."""
    _, mock_model = patch_st
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    t.transform(X)
    texts = mock_model.encode.call_args[0][0]
    assert texts == expected_texts


# ── Encode parameter forwarding ──────────────────────────────────────────────


def test_encode_parameters_forwarded(patch_st):
    """All configurable parameters are forwarded to model.encode()."""
    _, mock_model = patch_st
    t = SentenceEmbeddingTransformer(
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        precision="int8",
        prompt_name="query",
        prompt="Search: ",
    )
    t.fit(["a"])
    t.transform(["hello"])
    _, kwargs = mock_model.encode.call_args
    assert kwargs["batch_size"] == 64
    assert kwargs["show_progress_bar"] is True
    assert kwargs["convert_to_numpy"] is True  # always hardcoded
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["precision"] == "int8"
    assert kwargs["prompt_name"] == "query"
    assert kwargs["prompt"] == "Search: "


# ── get_feature_names_out ─────────────────────────────────────────────────────


def test_get_feature_names_out(patch_st):
    """Feature names follow dim_0, dim_1, ... pattern."""
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    names = t.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    expected = [f"dim_{i}" for i in range(EMBEDDING_DIM)]
    assert list(names) == expected


def test_get_feature_names_out_with_input_features(patch_st):
    """get_feature_names_out accepts input_features if length matches n_features_in_."""
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    names = t.get_feature_names_out(["text_col"])
    expected = [f"dim_{i}" for i in range(EMBEDDING_DIM)]
    assert list(names) == expected


def test_get_feature_names_out_invalid_input_features(patch_st):
    """get_feature_names_out raises ValueError if input_features length is incorrect."""
    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    with pytest.raises(ValueError, match="expected 1"):
        t.get_feature_names_out(["col1", "col2"])


def test_get_feature_names_out_before_fit():
    """Feature names before fit() raises NotFittedError."""
    t = SentenceEmbeddingTransformer()
    with pytest.raises(NotFittedError):
        t.get_feature_names_out()


# ── Pipeline integration ─────────────────────────────────────────────────────


def test_pipeline_fit_transform(patch_st):
    """SentenceEmbeddingTransformer works inside an sklearn Pipeline."""
    pipe = Pipeline([("embed", SentenceEmbeddingTransformer())])
    out = pipe.fit_transform(["a", "b", "c"])
    assert out.shape == (3, EMBEDDING_DIM)


def test_pipeline_pandas_output(patch_st):
    """Pipeline with set_output uses get_feature_names_out for columns."""
    pipe = Pipeline([("embed", SentenceEmbeddingTransformer())])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(["x", "y"])
    assert isinstance(df, pd.DataFrame)
    expected_cols = [f"dim_{i}" for i in range(EMBEDDING_DIM)]
    assert list(df.columns) == expected_cols


# ── fit_transform shorthand ──────────────────────────────────────────────────


def test_fit_transform(patch_st):
    """Fit_transform works as a shorthand for fit() then transform()."""
    t = SentenceEmbeddingTransformer()
    out = t.fit_transform(["hello", "world"])
    assert out.shape == (2, EMBEDDING_DIM)


# ── sklearn clone / get_params ────────────────────────────────────────────────


def test_get_params():
    """Get_params returns all constructor parameters."""
    t = SentenceEmbeddingTransformer(model_name="test", batch_size=16)
    params = t.get_params()
    assert params["model_name"] == "test"
    assert params["batch_size"] == 16
    assert params["show_progress_bar"] is False


def test_set_params():
    """Set_params updates parameters correctly."""
    t = SentenceEmbeddingTransformer()
    t.set_params(batch_size=128, precision="int8")
    assert t.batch_size == 128
    assert t.precision == "int8"


def test_set_params_invalid_precision():
    """Setting an invalid precision via set_params raises ValueError."""
    # No patch_st needed since set_params raises before loading the model.
    t = SentenceEmbeddingTransformer()
    with pytest.raises(ValueError, match="Invalid precision 'invalid_prec'"):
        t.set_params(precision="invalid_prec")


# ── Empty input ──────────────────────────────────────────────────────────────


def test_empty_input_list(patch_st):
    """Transforming an empty list returns (0, embedding_dim) array."""
    _, mock_model = patch_st
    original_side_effect = mock_model.encode.side_effect

    def _encode_empty(sentences, **kwargs):
        if len(sentences) == 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return original_side_effect(sentences, **kwargs)

    mock_model.encode.side_effect = _encode_empty

    t = SentenceEmbeddingTransformer()
    t.fit(["a"])
    out = t.transform([])
    assert out.shape == (0, EMBEDDING_DIM)
