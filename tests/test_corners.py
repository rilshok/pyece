from pytest import mark
from pyece.corners import Corners

@mark.parametrize(
    "shape",
    [
        (1,2),
        (1,2,3),
        (1,2,3,4)
    ],
)
def test_corners_product(shape):
    cs = Corners.product(shape)
    assert cs.dim == len(shape)
    assert cs.centre.mean() == sum(shape)/len(shape)/2
