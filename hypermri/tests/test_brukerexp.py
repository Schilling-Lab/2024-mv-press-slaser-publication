import pytest
from hypermri import BrukerExp


def test_BrukerExp():
    """Small dummy test."""

    with pytest.raises(FileNotFoundError, match=r"Folder"):
        scan = BrukerExp("./tests/data")
