from pytest import mark
from pyece.corners import Corners


@mark.parametrize(
    "shape",
    [(1, 2), (1, 2, 3), (1, 2, 3, 4)],
)
def test_corners_product(shape):
    cs = Corners.product(shape).value
    assert cs.shape[0] == 2**len(shape)
    assert cs.mean() == sum(shape) / len(shape) / 2
