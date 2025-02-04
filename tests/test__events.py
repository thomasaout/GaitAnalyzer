import pytest
import numpy.testing as npt


@pytest.mark.parametrize("a", [1.0, 1.0])
def test_dummy(a):
    # Example of test to perform
    npt.assert_almost_equal(a, 1.0)
    return
