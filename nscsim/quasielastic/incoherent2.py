import numpy as np
import functools
from nscsim.utilities import glog, shared_array
from nscsim import qvec
from tqdm import tqdm
import os
import pathos


def intermediate_vector_set(tr, q, s_inc, n_cores=None):
    r"""
    Incoherent scattering for a collection of vectors.

    We sum the intensities math:`A_i` derived for each atom math:`i`.
    The intensities math:`A_i` are self-correlations of atomic
    amplitudes. The amplitudes math:`a_i` are complex numbers.
    math:`a_i(\vec{q}, t) = e^{-i\vec{q}\vec(r)_i(t)}
    math: `A_i(\vec{q}, t) = si * <a_i^*(\vec{q}, t_0) a_i^*(\vec{q}, t_0+t)>_t_0`

    Parameters
    ----------
    tr: numpy.ndarrary
        Atomic trajectory, shape=(#atoms, #frames, 3)
    q: numpy.ndarray
        Array of q-vectors shape=(#vectors, 3)
    s_inc: numpy.ndarray
        Incoherent scattering cross-section, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core

    Returns
    -------
    np.ndarray
        Scattering atomic intensities shape=(#q's, 2 * #frames - 1)
    """

    def serial_worker(atomic_tr, atomic_s_inc):
        r"""
        Calculate intensities for a single atom

        Parameters
        ----------
        atomic_tr: numpy.ndarray
            shape = (#frames, 3) trajectory for one atom
        atomic_s_inc: float
            Incoherent scattering cross-section for one atom
        Returns
        -------
        numpy.ndarray
            shape = (#q's, 2 * #frames - 1)
        """
        r"""Pseudocode:
        - calculate a_i(\vec{q}, t)
        - return self-correlation of a_i to obtain A_i(\vec{q}, t)
        """
        ai = np.exp(-1j * np.tensordot(q, atomic_tr, axes=(1, 1)))
        result = np.apply_along_axis(
            lambda x: atomic_s_inc*np.correlate(x, x, "full"),
            1,
            ai,
        )
        return result

    glog.info("\nCalculating incoherent intensities for one set of q vectors\n")
    # TODO: close_pool=False is a temporary fix. See issue #31
    #
    # Problem: variable amps with shape=(#atoms, #q, 2*#frame-1)
    #          is too big to fit in memory. We have to add results
    #          of map_parallel to a variable sum_amps as soon as
    #          each core terminates serial_worker.
    #
    pool = pathos.pools.ProcessPool(nodes=n_cores)
    glog.info(f"\nUsing pool of {pool.nodes} cpus.\n")
    try:
        pool.restart(force=True)
        result = functools.reduce(
            np.add,
            pool.uimap(serial_worker, shared_array(tr), s_inc),
        )  # shape = (#q's, 2*#frames - 1)
    finally:
        pool.close()
        pool.join()
    return result


def intermediate(
    tr, q, s_inc, n_cores=None, averaging=(np.average, None, dict(axis=0))
):
    r"""
    Incoherent scattering averaged over scattering for a collection of q-vectors

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#atoms, #frames, 3)
    q: numpy.ndarray
        Array of q-vectors shape=(#vectors, 3)
    s_inc: numpy.ndarray
        Incoherent scattering cross-section, shape=(#atoms, 1)
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
    sf = intermediate_vector_set(tr, q, s_inc, n_cores=n_cores)
    av_args = list() if averaging[1] is None else averaging[1]
    return averaging[0](sf, *av_args, **averaging[2])


def intermediate_spherical(tr, q_mod, s_inc, n_cores=None):
    r"""
    Coherent scattering averaged over scattering for a collection of q-vectors

    Parameters
    ----------
    tr: numpy.ndarray
        Atomic trajectory, shape=(#atoms, #frames, 3)
    q_mod: numpy.ndarray
        Array of q-vector moduli shape=(#q-moduli, 1)
    s_inc: numpy.ndarray
        Incoherent scattering cross-section, shape=(#atoms, 1)
    n_cores: int
        Number of CPU's to use. All cores minus 1, or 1 if only 1 core

    Returns
    -------
    np.ndarray
        shape = (#frames)
    """
    q_sets = qvec.sphere_average(q_mod)
    glog.info("\nCalculating coherent scattering for a set of q-moduli\n")
    sf = [
        np.real(intermediate(tr, q_set, s_inc, n_cores=n_cores))
        for q_set in tqdm(q_sets)
    ]
    return np.asarray(sf)
