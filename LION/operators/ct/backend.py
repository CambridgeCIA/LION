# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The library that computes LION's CT projections: tomosipo (with astra) or
TIGRE.

Pick it once, at the top of a script, and LION follows it (like the device)::

    from LION.operators.ct import set_ct_backend
    set_ct_backend("tigre")

A geometry's ``backend`` (``"default"`` unless set) chooses it for one
operator; ``"default"`` is replaced by the backend used when the operator is
made, so records say which one made a result and loading uses it again.
"""

from __future__ import annotations

from LION.utils.parameter import DEFAULT_BACKEND

#: The CT backends LION can use.
CT_BACKENDS = ("tomosipo", "tigre")

_ct_backend = "tomosipo"


def check_ct_backend(backend: str) -> str:
    """``backend``, or a ValueError if it isn't one of ``CT_BACKENDS``."""
    if backend not in CT_BACKENDS:
        raise ValueError(
            f"CT backend {backend!r} not supported; it must be one of {CT_BACKENDS}"
        )
    return backend


def set_ct_backend(backend: str) -> None:
    """Make ``backend`` ("tomosipo" or "tigre") the one CT operators use when
    their geometry doesn't choose one (``backend = "default"``)."""
    global _ct_backend
    _ct_backend = check_ct_backend(backend)


def ct_backend() -> str:
    """The backend CT operators use when their geometry doesn't choose one."""
    return _ct_backend


def resolve_ct_backend(backend: str | None) -> str:
    """``backend``; for ``"default"`` (or None), the one set with
    ``set_ct_backend``."""
    if backend is None or backend == DEFAULT_BACKEND:
        return ct_backend()
    return check_ct_backend(backend)


def require_ct_backend(op, backend: str, feature: str) -> None:
    """A NotImplementedError if the CT operator ``op`` is computed by another
    backend than ``backend``, which ``feature`` (e.g. "sirt") needs."""
    if getattr(op, "backend", None) not in (None, backend):
        raise NotImplementedError(
            f"{feature} needs the {backend} backend; this operator's is "
            f"{op.backend!r}. Make it with backend={backend!r} (the geometry's "
            "backend, or set_ct_backend)."
        )
