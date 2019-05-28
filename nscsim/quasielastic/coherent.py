from __future__ import (print_function, absolute_import)

import numpy as np
from nscsim.utilities import (glog, map_parallel, shared_array)
from nscsim import qvec
from tqdm import tqdm


def intermediate_amplitudes(tr, q, bc, n_cores=None):
    r"""
    For every frame, calculate the coherent amplitude.

    The coherent amplitude is a complex number
    math:`A(\vec{q}) = sum_i b_i e^{-i\vec{q}\vec(r)_i}

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#frames, #atoms, 3)
    q: numpy.ndarray
        Array of q-vectors shape=(#vectors, 3)
    bc: numpy.ndarray
        Coherent scattering lengths, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core

    Returns
    -------
    np.ndarray
        Scattering amplitudes shape=(#frames, #q's)
    """

    def serial_worker(frame):
        r"""
        Calculate q-dependent coherent amplitudes for one frame

        Parameters
        ----------
        frame: numpy.ndarray
            shape = (#atoms, 3) coordinates of the system
        """
        exponents = np.tensordot(frame, q, axes=(1, 1))  # shape=(#atoms, #q's)
        exponentials = np.exp(1j * exponents)
        return np.tensordot(bc, exponentials, axes=1)  # shape=(#q's,)

    glog.info('\nCalculating coherent amplitudes for one set of q vectors\n')
    #TODO: close_pool=False is a temporary fix. See issue #31
    amps = np.array(map_parallel(serial_worker, shared_array(tr), n_cores, close_pool=False))

    return amps


def times(tr):
    r"""
    Times for the coherent scattering functions. Units are whatever the
    time step between frames in the input trajectory.

    Minimum time is `1-len(tr)` and maximum time is `len(tr)-1`

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#frames, #atoms, 3)ndarray

    Returns
    -------
    numpy.ndarray
    """
    return np.arange(1 - len(tr), len(tr))


def intermediate_vector_set(tr, q, bc, n_cores=None):
    r"""
    Coherent scattering for a collection of vectors

    Coherent scattering is reported for each vector in the set.
    The scattering curve is symmetrized,
    e.g. if the trajectory has 10 frames
    then the scattering curve would start at t=-9
    and runs up to t=9.


    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#frames, #atoms, 3)
    q: numpy.ndarray
        Array of q-vectors shape=(#vectors, 3)
    bc: numpy.ndarray
        Coherent scattering lengths, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core

    Returns
    -------
    np.ndarray
        shape = (len(q), 2 * #frames - 1)
    """
    amps = intermediate_amplitudes(tr, q, bc, n_cores=n_cores).transpose()
    w = np.arange(1, 1 + len(tr))
    w = 1.0 / np.concatenate((w, w[::-1][1:]))
    sf = [w * np.correlate(a, a, 'full') for a in amps]
    return np.asarray(sf)


def intermediate(tr, q, bc, n_cores=None,
                 averaging=(np.average, None, dict(axis=0))):
    r"""
    Coherent scattering averaged over scattering for a collection of q-vectors

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#frames, #atoms, 3)
    q: numpy.ndarray
        Array of q-vectors shape=(#vectors, 3)
    bc: numpy.ndarray
        Coherent scattering lengths, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core
    averaging: tuple
        The Q-average to carry out over the :math:`S(\vec{q}, t)`, the
        return value from `intermediate_vector_set`.
        Tuple elements:
            - averaging function
            - positional arguments for function, excluded the first which is
              assumed to be array :math:`S(\vec{q}, t)`
            - keyword arguments for function

    Returns
    -------
    np.ndarray
        shape = (#frames)
    """
    sf = intermediate_vector_set(tr, q, bc, n_cores=n_cores)
    av_args = list() if averaging[1] is None else averaging[1]
    return averaging[0](sf, *av_args, **averaging[2])


def intermediate_spherical(tr, q_mod, b_c, n_cores=1):
    r"""
    Coherent scattering averaged over scattering for a collection of q-vectors

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#frames, #atoms, 3)
    q_mod: numpy.ndarray
        Array of q-vector moduli shape=(#q-moduli, 1)
    b_c: numpy.ndarray
        Coherent scattering lengths, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core

    Returns
    -------
    np.ndarray
        shape = (#frames)
    """
    q_sets = qvec.sphere_average(q_mod)
    glog.info('\nCalculating coherent scattering for a set of q-moduli\n')
    sf = [np.real(intermediate(tr, q_set, b_c, n_cores=n_cores))
          for q_set in tqdm(q_sets)]
    return np.asarray(sf)
