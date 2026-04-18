"""Common utilities for transformers module."""

import re
from typing import Any


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
