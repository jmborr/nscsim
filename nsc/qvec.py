from __future__ import (print_function, absolute_import)

import numpy as np


def unit_sphere_surf(nvec=200):
    """
    Vectors randomly sampling the surface of the unit sphere

    Parameters
    ----------
    nvec: int
        Number of vectors to generate. Must be an even number

    Returns
    -------
    numpy.ndarray
        Array of vectors, shape = (nvec, 3)
    """
    if nvec < 2 or nvec % 2 != 0:
        raise ValueError('number of vectors must be an even'
                         ' number equal to or bigger than 2')
    v = np.random.normal(size=(nvec, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def moduli(q_mod_min, q_mod_max=None, q_mod_delta=None,
           n_q_mod=None):
    """
    An array of q vector moduli (q_mods) is generated depending on the passed
    arguments. These are the options:
    1. (q_mod_min, q_mod_max, q_mod_delta) generates q_mods
    2. (q_mod_min, q_mod_max, n_q_mod) generates q_mods
    3. (q_mod_min, q_mod_delta, n_q_mod) generates q_mods

    Parameters
    ----------
    q_mod_min: float
        Minimum q-vector modulus
    q_mod_max: float
        Maximum q-vector modulus
    q_mod_delta: float
        Increase in q-vector modulus
    n_q_mod: int
        Number of q-vector moduli

    Returns
    -------
    numpy.ndarray
        Array of q-vector moduli
    """
    q_mods = None

    if q_mod_max is not None:
        if q_mod_delta is not None:
            # q_mods from triad (q_mod_min, q_mod_max, q_mod_delta)
            q_mods = np.arange(q_mod_min, q_mod_max, q_mod_delta)
        elif n_q_mod is not None:
            # q_mods from triad (q_mod_min, q_mod_max, n_q_mod)
            q_mod_delta = (q_mod_max - q_mod_min) / n_q_mod
            q_mods = np.arange(q_mod_min, q_mod_max, q_mod_delta)
    elif q_mod_delta is not None and n_q_mod is not None:
        # q_mods from triad (q_mod_min, q_mod_delta, n_q_mod)
        q_mods = q_mod_min + q_mod_delta * np.arange(n_q_mod)

    if q_mods is None:
        raise ValueError('Insufficient arguments passed.'
                         ' No q-vector moduli generated')
    return q_mods


def sphere_average(q_mod_array=None, q_mod_min=None, q_mod_max=None,
                   q_mod_delta=None, n_q_mod=None, nvec=200):
    """
    Set of q-vectors with random orientations and of specified moduli.

    An array of q vector moduli (q_mods) will be used that will depend
    on the passed arguments. These are the options:
    1. q_mod_array is referenced by q_mods
    2. (q_mod_min, q_mod_max, q_mod_delta) generates q_mods
    3. (q_mod_min, q_mod_max, n_q_mod) generates q_mods
    4. (q_mod_min, q_mod_delta, n_q_mod) generates q_mods

    Parameters
    ----------
    q_mod_array:
        Array of q-vector moduli (list, tuple, numpy.ndarray, iterator..)
    q_mod_min: float
        Minimum q-vector modulus
    q_mod_max: float
        Maximum q-vector modulus
    q_mod_delta: float
        Increase in q-vector modulus
    n_q_mod: int
        Number of q-vector moduli
    nvec: int
        Number of orientations. Must be an even number
    Returns
    -------
    numpy.ndarray
        Array of vectors, shape = (len(q_mods), nvec, 3)
    """

    if q_mod_array is not None:
        q_mods = np.asarray(q_mod_array)
    else:
        q_mods = moduli(q_mod_min, q_mod_max,
                        q_mod_delta, n_q_mod)

    return np.tensordot(q_mods, unit_sphere_surf(nvec), axes=0)
