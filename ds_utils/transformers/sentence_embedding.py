"""Scikit-learn compatible transformer for sentence-transformers embeddings.

Requires the optional ``nlp`` dependency group::

    pip install data-science-utils[nlp]
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, Sequence[Any]]


def _check_sentence_transformers_installed() -> None:
    """Raise a helpful ``ImportError`` if ``sentence-transformers`` is not installed."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for SentenceEmbeddingTransformer. "
            "Install it with:  pip install data-science-utils[nlp]"
        ) from exc


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Wrap a ``sentence-transformers`` model for use in sklearn pipelines.

    Loads a `SentenceTransformer
    <https://sbert.net/docs/package_reference/sentence_transformer/model.html>`_
    model lazily on first :meth:`fit` and produces a dense ``float32`` (or
    quantized) embedding matrix from text inputs.

    The transformer accepts strings, lists of strings, :class:`pandas.Series`,
    :class:`pandas.DataFrame` (single column), and :class:`numpy.ndarray`.
    ``None`` and ``NaN`` values are replaced with empty strings before encoding.

    .. note::

       This transformer requires the optional ``nlp`` extras::

           pip install data-science-utils[nlp]

    :param model_name: Name or path of a ``sentence-transformers`` model
        (default: ``'sentence-transformers/all-MiniLM-L6-v2'``).
    :param batch_size: Batch size for encoding (default: ``32``).
    :param show_progress_bar: Whether to show a progress bar during encoding
        (default: ``False``).
    :param normalize_embeddings: Whether to L2-normalize embeddings to unit
        length (default: ``False``).
    :param device: Device for computation (``'cpu'``, ``'cuda'``, etc.).
        ``None`` lets the library auto-detect (default: ``None``).
    :param precision: Embedding precision — ``'float32'``, ``'int8'``,
        ``'uint8'``, ``'binary'``, or ``'ubinary'`` (default: ``'float32'``).
    :param truncate_dim: Truncate embeddings to this many dimensions. Useful
        for `Matryoshka <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_
        models (default: ``None`` — no truncation).
    :param prompt_name: Name of a prompt registered in the model's
        ``prompts`` dictionary (default: ``None``).
    :param prompt: Raw prompt string to prepend to every input sentence
        (default: ``None``).

    :ivar model_: The loaded ``SentenceTransformer`` instance (set after :meth:`fit`).
    :ivar embedding_dimension_: Dimensionality of the output embeddings (set after :meth:`fit`).
    :ivar n_features_in_: Number of input features (always ``1``).
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        device: Optional[str] = None,
        precision: str = "float32",
        truncate_dim: Optional[int] = None,
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """See class docstring for parameter descriptions."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.precision = precision
        self.truncate_dim = truncate_dim
        self.prompt_name = prompt_name
        self.prompt = prompt

    def _load_model(self) -> None:
        """Import and instantiate the ``SentenceTransformer`` model."""
        _check_sentence_transformers_installed()
        from sentence_transformers import SentenceTransformer

        self.model_ = SentenceTransformer(self.model_name, device=self.device, truncate_dim=self.truncate_dim)
        self.embedding_dimension_ = self.model_.get_sentence_embedding_dimension()

    @staticmethod
    def _prepare_texts(X: ArrayLike) -> List[str]:
        """Convert input to a list of strings, replacing ``None``/``NaN`` with ``""``."""
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    f"SentenceEmbeddingTransformer expects a single text column; got DataFrame with shape {X.shape}."
                )
            raw = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            raw = X.tolist()
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                if X.shape[1] != 1:
                    raise ValueError(
                        f"SentenceEmbeddingTransformer expects a single text column; got array with shape {X.shape}."
                    )
                raw = X[:, 0].tolist()
            else:
                raw = X.tolist()
        elif isinstance(X, str):
            raw = [X]
        else:
            raw = list(X)

        return [str(t) if t is not None and not (isinstance(t, float) and pd.isna(t)) else "" for t in raw]

    def fit(self, X: ArrayLike, y: Any = None) -> SentenceEmbeddingTransformer:
        """Load the sentence-transformer model and record embedding metadata.

        The model is loaded lazily on the first call to :meth:`fit`. Subsequent
        calls reuse the cached model unless the transformer is re-created.

        :param X: Text data — array-like of strings, :class:`pandas.Series`,
            single-column :class:`pandas.DataFrame`, or :class:`numpy.ndarray`.
        :param y: Ignored; present for sklearn API compatibility.
        :return: This estimator, fitted.
        """
        self.n_features_in_ = 1
        if not hasattr(self, "model_"):
            self._load_model()
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode text inputs into dense embedding vectors.

        :param X: Same accepted forms as :meth:`fit`.
        :return: Embedding matrix of shape ``(n_samples, embedding_dimension_)``.
        :raises sklearn.exceptions.NotFittedError: If :meth:`fit` has not been called.
        """
        check_is_fitted(self, "model_")
        texts = self._prepare_texts(X)
        embeddings = self.model_.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            precision=self.precision,
            prompt_name=self.prompt_name,
            prompt=self.prompt,
        )
        return np.asarray(embeddings)

    def get_feature_names_out(self, input_features: Union[None, np.ndarray, List[str]] = None) -> np.ndarray:
        """Return output feature names for this transformation.

        Names follow ``dim_0``, ``dim_1``, …, ``dim_{n-1}``.

        :param input_features: Ignored; present for sklearn API compatibility.
        :return: ``numpy.ndarray`` of shape ``(embedding_dimension_,)``, dtype ``object``.
        """
        check_is_fitted(self, "embedding_dimension_")
        return np.asarray([f"dim_{i}" for i in range(self.embedding_dimension_)], dtype=object)
