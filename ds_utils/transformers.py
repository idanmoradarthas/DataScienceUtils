"""Scikit-learn compatible transformer wrappers for preprocessing pipelines."""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, Sequence[Any]]


def _sanitize_column_name(name: Any) -> str:
    """Sanitize a label for use in feature names (e.g. Delta-safe identifiers).

    Replaces invalid characters (space, comma, semicolon, braces, parentheses,
    newline, tab, equals) with underscores, collapses repeated underscores, and
    strips leading and trailing underscores.

    :param name: Label or value from ``classes_``; coerced to ``str``.
    :return: Sanitized string suitable as part of a column name.
    """
    name_str = str(name)
    sanitized = re.sub(r"[ ,;{}()\n\t=]", "_", name_str)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """Wrap ``sklearn.preprocessing.MultiLabelBinarizer`` for sklearn pipelines.

    Learns a binary indicator matrix for multi-label data. Unlike using
    ``MultiLabelBinarizer`` alone, this class implements ``get_feature_names_out``
    (feature names API, SLEP007) and returns dense ``float64`` output for downstream
    steps.

    Pass one iterable of labels per sample. A flat list of strings is invalid:
    scikit-learn would treat each character as a sample. See
    `MultiLabelBinarizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html>`_.

    :param classes: Optional fixed ordering of class labels (passed to ``MultiLabelBinarizer``).
    :param sparse_output: If True, the inner binarizer may use sparse storage; :meth:`transform`
        still returns a dense ``float64`` ndarray.

    :ivar mlb_: Fitted ``MultiLabelBinarizer`` instance (set after :meth:`fit`).
    :ivar n_features_in_: Number of input features (always ``1``: one multi-label column).
    """

    def __init__(self, *, classes: Union[None, np.ndarray, Sequence[Any]] = None, sparse_output: bool = False) -> None:
        """See class docstring for ``classes`` and ``sparse_output``."""
        self.classes = classes
        self.sparse_output = sparse_output

    def _extract_column(self, X: ArrayLike) -> Sequence[Any]:
        """Return one sequence element per row (length ``n_samples``).

        Each element is one cell: an iterable of labels, a scalar label, or empty.

        :param X: DataFrame (single column), Series, ndarray, or sequence of per-row values.
        :return: One-dimensional sequence of row values to pass to :meth:`_row_to_labels`.
        :raise ValueError: If ``X`` is a DataFrame or 2D array with more than one feature column
            when a single multi-label column is required.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "MultiLabelBinarizerTransformer expects a single column (one multi-label "
                    f"feature); got shape {X.shape}."
                )
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr[:, 0]
            # Wide 2D ndarray: treat each row as one sample, columns as label entries.
            # Note: DataFrames with multiple columns are rejected (see ValueError above);
            # for ndarrays we allow this layout intentionally for numpy callers.
            return [row for row in arr]
        return arr

    def _row_to_labels(self, item: Any) -> List[Any]:
        """Convert one cell to a list of hashable labels for ``MultiLabelBinarizer``.

        :param item: One row value (list, set, tuple, ndarray, scalar, None, or NaN).
        :return: List of labels for that row; empty if missing or unparseable.
        """
        if item is None:
            return []
        if isinstance(item, float) and pd.isna(item):
            return []
        # Flatten all ndarrays to a Python list first, then re-enter via the list branch.
        if isinstance(item, np.ndarray):
            return self._row_to_labels(item.flatten().tolist())
        if isinstance(item, (list, tuple, set)):
            cleaned: List[Any] = []
            for x in item:
                if isinstance(x, np.ndarray):
                    # A nested ndarray element: unpack to a scalar or sub-list.
                    xi = x.item() if x.size == 1 else x.tolist()
                    if isinstance(xi, list):
                        cleaned.extend(
                            y
                            for y in xi
                            if isinstance(y, (str, int, float, bool, np.generic))
                            and not (isinstance(y, float) and pd.isna(y))
                        )
                    elif isinstance(xi, (str, int, float, bool, np.generic)) and not (
                        isinstance(xi, float) and pd.isna(xi)
                    ):
                        cleaned.append(xi)
                elif isinstance(x, (str, int, float, bool, np.generic)) and not (isinstance(x, float) and pd.isna(x)):
                    cleaned.append(x.item() if isinstance(x, np.generic) else x)
            return cleaned
        if isinstance(item, (str, int, float, bool, np.generic)) and not (isinstance(item, float) and pd.isna(item)):
            return [item.item() if isinstance(item, np.generic) else item]
        return []

    def _prepare(self, X: ArrayLike) -> List[List[Any]]:
        """Build the list-of-lists input expected by ``MultiLabelBinarizer``.

        :param X: Same accepted forms as :meth:`fit`.
        :return: One list of labels per sample.
        """
        col = self._extract_column(X)
        if hasattr(col, "tolist"):
            col_list = col.tolist()
        else:
            col_list = list(col)
        return [self._row_to_labels(row) for row in col_list]

    def fit(self, X: ArrayLike, y: Any = None) -> MultiLabelBinarizerTransformer:
        """Learn label sets from training multi-label data.

        :param X: Array-like of shape ``(n_samples,)`` or ``(n_samples, 1)``, or a wide 2D layout
            where each row is one sample and each column entry is a label for that sample.
        :param y: Ignored; present for sklearn API compatibility.
        :return: This estimator, fitted.
        """
        self._extract_column(X)  # validate shape/column-count before mutating state
        self.n_features_in_ = 1
        processed_X = self._prepare(X)
        self.mlb_ = MultiLabelBinarizer(classes=self.classes, sparse_output=self.sparse_output)
        self.mlb_.fit(processed_X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Binarize multi-label data using the vocabulary learned in :meth:`fit`.

        :param X: Same layout as for :meth:`fit`.
        :return: Binary indicator matrix of shape ``(n_samples, n_classes)``, dtype ``float64``.
        """
        check_is_fitted(self, "mlb_")
        self._extract_column(X)
        processed_X = self._prepare(X)
        result = self.mlb_.transform(processed_X)
        if hasattr(result, "toarray"):
            result = result.toarray()
        return np.asarray(result, dtype=np.float64)

    def get_feature_names_out(self, input_features: Union[None, np.ndarray, List[str]] = None) -> np.ndarray:
        """Return output feature names for this transformation.

        Names follow ``{prefix}_{sanitized_class}``. If ``input_features`` is omitted, the
        prefix is ``"label"``; otherwise the prefix is the first validated input feature name.

        :param input_features: Names for the input column(s), or None. When provided, length must
            match ``n_features_in_``.
        :return: ``numpy.ndarray`` of shape ``(n_classes,)``, dtype ``object``, of output names.
        """
        check_is_fitted(self, "mlb_")
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    f"input_features has {len(input_features)} element(s), expected {self.n_features_in_}."
                )
            prefix = str(input_features[0])
        else:
            prefix = "label"
        sanitized_labels = [_sanitize_column_name(c) for c in self.mlb_.classes_]
        out = [f"{prefix}_{lab}" for lab in sanitized_labels]
        return np.asarray(out, dtype=object)
