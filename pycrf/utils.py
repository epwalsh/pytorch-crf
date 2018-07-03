"""Helpers funcs."""

from typing import Tuple, Union


def _parse_data_path(path: str) -> Tuple[Union[str, None], str]:
    separated = path.split(":")
    if len(separated) == 1:
        return None, path
    if len(separated) == 2:
        return separated[0], separated[1]
    raise ValueError("Invalid path; extra ':' found.")
