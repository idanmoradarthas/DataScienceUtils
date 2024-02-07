from pathlib import Path


__version__ = Path(__file__).absolute().parents[1].joinpath(".version").read_text()
