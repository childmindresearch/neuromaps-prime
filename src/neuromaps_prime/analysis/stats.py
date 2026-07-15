"""Core statistical engine.

Provides functions for computing correlations, statistical tests, and
null-distribution metrics over pairs of neuroimaging maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = []


def _chk2_asarray(
    a: ArrayLike,
    b: ArrayLike,
    axis: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """Validate and convert two input sequences into NumPy arrays.

    When *axis* is ``None``, both inputs are flattened to 1-D and the
    output axis is set to ``0``.  Otherwise the inputs are kept in their
    original shape.  Scalar-like inputs are always promoted to at least
    1-D arrays.

    This function was part of the (now removed) private API of scipy
    (https://github.com/scipy/scipy/pull/23088).

    Parameters
    ----------
    a : ArrayLike
        First input array or array-like object.
    b : ArrayLike
        Second input array or array-like object.
    axis : int or None, optional
        Axis along which to operate.  If ``None`` (default), both inputs
        are flattened.

    Returns:
    --------
    tuple[NDArray, NDArray, int]
        Normalized ``(a, b, out_axis)`` where both arrays are at least
        1-D and ``out_axis`` is the axis to pass to downstream
        operations.

    Notes:
    ------
    Lifted from the legacy neuromaps codebase
    (https://github.com/netneurolab/neuromaps/blob/ffcc2e0f657943ce00a1b6a968396f32250e495c/neuromaps/utils.py#L131).
    """
    out_axis: int

    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        out_axis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        out_axis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)
    if b.ndim == 0:
        b = np.atleast_1d(b)

    return a, b, out_axis
