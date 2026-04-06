"""Tests for sklearn-compatible transformers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from ds_utils.transformers import MultiLabelBinarizerTransformer


def test_fit_transform_list_of_lists_basic():
    """List-of-lists input produces float64 binary matrix."""
    X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform(X)
    assert out.dtype == np.float64
    assert out.shape == (3, 4)
    expected = (
        MultiLabelBinarizer()
        .fit([["sci-fi", "action"], ["romance"], ["action", "comedy"]])
        .transform([["sci-fi", "action"], ["romance"], ["action", "comedy"]])
    )
    np.testing.assert_array_equal(out, expected.astype(np.float64))


def test_get_feature_names_out_default_prefix():
    """Default prefix is label; names match sanitized classes order."""
    X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
    mlb = MultiLabelBinarizerTransformer()
    mlb.fit(X)
    names = mlb.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert list(names) == ["label_action", "label_comedy", "label_romance", "label_sci-fi"]


def test_get_feature_names_out_custom_prefix_via_input_features():
    """First validated input feature name becomes prefix."""
    X = [["a", "b"], ["c"]]
    mlb = MultiLabelBinarizerTransformer()
    mlb.fit(X)
    names = mlb.get_feature_names_out(input_features=["my_col"])
    assert list(names) == ["my_col_a", "my_col_b", "my_col_c"]


def test_get_feature_names_out_wrong_input_features_length():
    """Passing more than one input feature name raises ValueError."""
    X = [["a", "b"], ["c"]]
    mlb = MultiLabelBinarizerTransformer()
    mlb.fit(X)
    with pytest.raises(ValueError, match="input_features has"):
        mlb.get_feature_names_out(input_features=["col_a", "col_b"])


def test_get_feature_names_out_sanitizes_invalid_chars():
    """Labels with spaces and punctuation are sanitized in feature names."""
    X = [["a b", "c,d"]]
    mlb = MultiLabelBinarizerTransformer()
    mlb.fit(X)
    names = mlb.get_feature_names_out()
    assert list(names) == ["label_a_b", "label_c_d"]


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(np.array([["x", "y"], ["z"], []], dtype=object), id="1d_object_array"),
        pytest.param(np.array([[["a", "b"]], [["c"]], [[]]], dtype=object), id="2d_single_column"),
    ],
)
def test_numpy_object_array(X):
    """Test numpy array of object (lists per row or 2D single column)."""
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform(X)
    assert out.shape == (3, 3)
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    ("data", "expected_shape"),
    [
        pytest.param(pd.Series([["a", "b"], ["c"], None], dtype=object), (3, 3), id="pandas_series"),
        pytest.param(pd.DataFrame({"tags": [["a"], ["b", "c"]]}), (2, 3), id="pandas_dataframe"),
    ],
)
def test_pandas_inputs(data, expected_shape):
    """Accept pd.Series and single-column DataFrame."""
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform(data)
    assert out.dtype == np.float64
    assert out.shape == expected_shape


def test_pipeline_and_set_output_pandas():
    """Pipeline with pandas output uses get_feature_names_out for columns."""
    X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
    pipe = Pipeline([("mlb", MultiLabelBinarizerTransformer())])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(X)
    expected_cols = list(pipe.named_steps["mlb"].get_feature_names_out())
    assert list(df.columns) == expected_cols


def test_none_and_nan_cells_become_empty_labels():
    """None and NaN map to empty label sets."""
    X = pd.Series([["a"], None, np.nan], dtype=object)
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform(X)
    assert out.shape[0] == 3
    assert out[1].sum() == 0
    assert out[2].sum() == 0


def test_single_label_per_sample():
    """Single string label per row."""
    X = [["only"], ["two"], ["only"]]
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform(X)
    assert out.sum() == 3.0  # two positives for 'only', one for 'two'


def test_rejects_multi_column_dataframe():
    """More than one column raises."""
    df = pd.DataFrame({"a": [[1]], "b": [[2]]})
    mlb = MultiLabelBinarizerTransformer()
    with pytest.raises(ValueError, match="single column"):
        mlb.fit(df)


def test_iterable_of_iterables_not_flat_list():
    """Flat list of three strings would be wrong for sklearn MLB (char samples)."""
    wrong = ["sci-fi", "thriller", "comedy"]
    mlb_wrong = MultiLabelBinarizer()
    mlb_wrong.fit(wrong)
    # Our wrapper treats one string as one label in one sample -> one row
    mlb = MultiLabelBinarizerTransformer()
    out = mlb.fit_transform([wrong])  # one sample: iterable of three labels
    assert out.shape == (1, 3)
    assert mlb.get_feature_names_out().size == 3
    assert mlb_wrong.classes_.size > 10


@pytest.mark.parametrize(
    ("row", "expected_cols"),
    [
        pytest.param(
            np.array([np.array(["a", "b"]), np.array("c"), "d"], dtype=object),
            ["label_a", "label_b", "label_c", "label_d"],
            id="1d_ndarray_containing_ndarrays",
        ),
        pytest.param(
            (np.array(["e", "f"]), np.array("g"), "h"),
            ["label_e", "label_f", "label_g", "label_h"],
            id="tuple_containing_ndarrays",
        ),
        pytest.param(
            np.array([["i", "j"], ["k", "l"]], dtype=object),
            ["label_i", "label_j", "label_k", "label_l"],
            id="multidimensional_ndarray",
        ),
    ],
)
def test_row_to_labels_edge_cases(row, expected_cols):
    """Test _row_to_labels edge cases with parameterized inputs."""
    mlb = MultiLabelBinarizerTransformer()
    X = pd.Series([row])
    out = mlb.fit_transform(X)
    assert out.shape == (1, 4)
    assert list(mlb.get_feature_names_out()) == expected_cols


def test_row_to_labels_unknown_object():
    """Test _row_to_labels edge case: unknown object should return empty labels."""
    mlb = MultiLabelBinarizerTransformer()

    class Unknown:
        pass

    row = Unknown()
    X = pd.Series([row])
    out = mlb.fit_transform(X)
    assert out.shape == (1, 0)


def test_sparse_output():
    """Verify sparse_output argument is accepted but returns dense array."""
    mlb = MultiLabelBinarizerTransformer(sparse_output=True)
    X = [["a", "b"], ["c"]]
    out = mlb.fit_transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
