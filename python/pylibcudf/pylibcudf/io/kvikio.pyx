# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t
from libcpp.optional cimport make_optional, nullopt, optional
from pylibcudf.libcudf.io.config_utils cimport set_up_kvikio as cpp_set_up_kvikio

__all__ = ["set_up_kvikio"]


cpdef void set_up_kvikio(object nthreads=None):
    """Set KvikIO parameters.

    Parameters include:

    - Compatibility mode, according to the environment variable ``KVIKIO_COMPAT_MODE``. If
      ``KVIKIO_COMPAT_MODE`` is not set, enable it by default, which enforces the use of POSIX I/O.
    - Thread pool size. If ``nthreads`` is provided, it is used directly. Otherwise, the value is
      read from the environment variable ``KVIKIO_NTHREADS``, defaulting to 4 if unset.

    Parameters
    ----------
    nthreads : int, optional
        Thread pool size override. If provided, supersedes ``KVIKIO_NTHREADS``.

    Returns
    -------
    None
    """
    cdef optional[uint32_t] c_nthreads = nullopt
    if nthreads is not None:
        c_nthreads = make_optional[uint32_t](<uint32_t>nthreads)
    with nogil:
        cpp_set_up_kvikio(c_nthreads)
