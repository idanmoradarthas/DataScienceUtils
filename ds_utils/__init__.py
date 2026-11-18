"""Data Science Utilities package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("data_science_utils")
except PackageNotFoundError:  # pragma: no cover - source tree that is not installed
    __version__ = "unknown"
