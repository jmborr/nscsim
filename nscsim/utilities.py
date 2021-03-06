from __future__ import (absolute_import, division, print_function)

import logging
import functools
import numpy as np
from collections import namedtuple, Mapping
import pathos
import multiprocessing
import ctypes


glog = logging.getLogger('nscsim')
glog.addHandler(logging.StreamHandler())
glog.setLevel(logging.INFO)


def namedtuplefy(func):
    r"""
    Decorator to transform the return dictionary of a function into
    a namedtuple

    Parameters
    ----------
    func: Function
        Function to be decorated
    name: str
        Class name for the namedtuple. If None, the name of the function
        will be used
    Returns
    -------
    Function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if wrapper.nt is None:
            if isinstance(res, Mapping) is False:
                raise ValueError('Cannot namedtuplefy a non-dict')
            wrapper.nt = namedtuple(func.__name__ + '_nt', res.keys())
        return wrapper.nt(**res)
    wrapper.nt = None
    return wrapper


def shared_array(from_array=None, shape=None, c_type='double'):
    r"""
    Read-only array shared by all CPU's

    Parameters
    ----------
    from_array: numpy.ndarray
        Create a shared array by copying a non-shared array. will overwrite
        any values passed through arguments `shape` and `c_type`
    shape: list
        desired shape of the array
    c_type: str
        One of 'double', 'longdouble, 'float',... See
        https://docs.python.org/3/library/ctypes.html

    Returns
    -------
    shared numpy.ndarray
    """
    dtype_to_ctype = {'float64': 'double', 'float32': 'float'}
    if from_array is not None:
        shape = from_array.shape
        dt = str(from_array.dtype)
        c_type = dtype_to_ctype.get(dt, dt)
    ctypes_class = getattr(ctypes, 'c_' + c_type)
    _array_base = multiprocessing.Array(ctypes_class,
                                        int(np.prod(shape)),
                                        lock=False)
    _array = np.ctypeslib.as_array(_array_base)
    _array = _array.reshape(shape)
    if from_array is not None:
        _array[:] = from_array
    return _array


def map_parallel(worker, iterator, ncpus, close_pool=True):
    pool = pathos.pools.ProcessPool(ncpus=ncpus)
    try:
        work = pool.map(worker, iterator)
    finally:
        if close_pool is True:
            pool.terminate()
    return work
