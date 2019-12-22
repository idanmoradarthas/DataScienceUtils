import sys

import matplotlib

matplotlib.use('agg')
from matplotlib.testing.compare import compare_images


def compare_images_paths(first: str, second: str) -> None:
    results = compare_images(first, second, 10)
    sys.stderr.write(results)
    if results is not None:  # the images compare favorably
        assert False
