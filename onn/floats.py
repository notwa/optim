import numpy as np

_f = np.float32

def _check(a):
    assert isinstance(a, np.ndarray) or type(a) == _f, type(a)
    assert a.dtype == _f, a.dtype
    return a

_0 = _f(0)
_1 = _f(1)
_2 = _f(2)
_inv2 = _f(1/2)
_sqrt2 = _f(np.sqrt(2))
_invsqrt2 = _f(1/np.sqrt(2))
_pi = _f(np.pi)

__all__ = [o for o in locals()]
