import sys

from matplotlib.testing.compare import compare_images

TOLERANCE = 40


def compare_images_paths(first: str, second: str) -> None:
    results = compare_images(first, second, TOLERANCE)
    if results is not None:  # the images compare favorably
        sys.stderr.write(results)
        assert False
