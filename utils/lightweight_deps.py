"""
Lightweight pure-numpy replacements for heavy ML dependencies (sklearn, skimage)
that insightface imports but barely uses.

InsightFace only uses:
  - skimage.transform.SimilarityTransform (for face alignment)
  - sklearn.metrics.pairwise.cosine_similarity (for evaluation, not core inference)

This module provides drop-in replacements using only numpy, saving ~130MB of
installed packages (scipy, scikit-learn, scikit-image + their transitive deps).

Must be called BEFORE importing insightface.
"""

import sys
import types
import numpy as np


def _make_module(name, parent=None):
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    sys.modules[name] = mod
    if parent is not None and '.' in name:
        attr_name = name.split('.')[-1]
        setattr(parent, attr_name, mod)
    return mod


def _cosine_similarity(X, Y=None):
    """Drop-in replacement for sklearn.metrics.pairwise.cosine_similarity."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return X_norm @ Y_norm.T


class SimilarityTransform:
    """
    Drop-in replacement for skimage.transform.SimilarityTransform.
    Implements the Umeyama algorithm for estimating similarity transforms.
    
    Reference: Shinji Umeyama, "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns", IEEE TPAMI, 1991.
    """

    def __init__(self):
        self.params = np.eye(3, dtype=np.float64)

    def estimate(self, src, dst):
        """Estimate the similarity transform from src to dst point sets."""
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)

        num = src.shape[0]
        dim = src.shape[1]

        # Compute means
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract means
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38) — covariance matrix
        A = dst_demean.T @ src_demean / num

        # Eq. (39)
        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.float64)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43)
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            self.params = np.nan * T
            return False

        if rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        # Eq. (41) and (42) — with scale
        src_var = src_demean.var(axis=0).sum()
        if src_var < 1e-10:
            scale = 1.0
        else:
            scale = 1.0 / src_var * (S @ d)

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
        T[:dim, :dim] *= scale

        self.params = T
        return True


def install():
    """
    Install lightweight mock modules for sklearn and skimage.
    Only installs if the real packages are not already available.
    Safe to call multiple times.
    """
    # --- Mock sklearn (if not installed) ---
    if 'sklearn' not in sys.modules:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            sklearn_mod = _make_module('sklearn')
            metrics = _make_module('sklearn.metrics', sklearn_mod)
            pairwise = _make_module('sklearn.metrics.pairwise', metrics)
            pairwise.cosine_similarity = _cosine_similarity

            # Also mock sklearn.preprocessing (sometimes imported)
            preprocessing = _make_module('sklearn.preprocessing', sklearn_mod)
            preprocessing.normalize = lambda X, **kw: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

            print("[lightweight_deps] Installed sklearn mock (cosine_similarity)")

    # --- Mock skimage (if not installed) ---
    if 'skimage' not in sys.modules:
        try:
            import skimage  # noqa: F401
        except ImportError:
            skimage_mod = _make_module('skimage')
            transform = _make_module('skimage.transform', skimage_mod)
            transform.SimilarityTransform = SimilarityTransform

            # Mock other skimage submodules that insightface might try to import
            _make_module('skimage.io', skimage_mod)
            _make_module('skimage.feature', skimage_mod)

            print("[lightweight_deps] Installed skimage mock (SimilarityTransform)")

    # --- Mock scipy (if not installed) ---
    if 'scipy' not in sys.modules:
        try:
            import scipy  # noqa: F401
        except ImportError:
            scipy_mod = _make_module('scipy')
            _make_module('scipy.spatial', scipy_mod)
            _make_module('scipy.spatial.distance', scipy_mod)

            print("[lightweight_deps] Installed scipy mock")
