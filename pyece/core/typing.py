from typing import *
import nptyping as nptp

NDArray = nptp.NDArray
IntTuple = Union[NDArray[nptp.Shape["*"], nptp.Integer], Tuple[int, ...]]
FloatTuple = Union[NDArray[nptp.Shape["*"], nptp.Float], Tuple[float, ...]]
