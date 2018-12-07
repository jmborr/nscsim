from __future__ import (absolute_import, division, print_function)

import functools
from collections import namedtuple, Mapping


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
