from __future__ import (print_function, absolute_import)

import numpy as np
import multiprocessing
from tqdm import tqdm as progress_bar
import pathos

from nscsim.utilities import namedtuplefy


def si_vector_self_intermediate(q, tr, ns=200, nt=100, dt=1):
    r"""
    Single atoms self incoherent structure factor for each vector in
    a set of q-vectors

    Parameters
    ----------
    q: numpy.ndarray
        q vectors shape=(#vectors, 3) for a set of q-vectors or
        shape=(#sets, #vectors, 3) for a list of sets of q-vectors
    tr: numpy.ndarray
        atomic trajectory for one atom, shape=(#frames, 3)
    ns: int
        Number of t0's for a given t
    nt: int
        Number of t's (number time lapses)
    dt: int
        Spacing between consecutive t's, in number of frames

    Returns
    -------
    numpy.ndarray
        array of shape=(#vectors) and the structure factor is normalized
        by the number of frames
    """
    pass


def si_self_intermediate_si(q, tr, ns=200, nt=100, dt=1):
    r"""
    Single atom self incoherent structure factor for each vector in a set
    of q-vectors, or for each set average in a list of sets of q-vectors.

    Parameters
    ----------
    q: numpy.ndarray
        q vectors shape=(#vectors, 3) for a set of q-vectors or
        shape=(#sets, #vectors, 3) for a list of sets of q-vectors
    tr: numpy.ndarray
        atomic trajectory for one atom, shape=(#frames, 3)
    ns: int
        Number of t0's for a given t
    nt: int
        Number of t's (number time lapses)
    dt: int
        Spacing between consecutive t's, in number of frames

    Returns
    -------
    numpy.ndarray
        If q.shape=(#vectors, 3), returns shape=(#vectors) and the structure
        factor is normalized by the number of frames and the sum of the
        squares of scattering lengths. If q.shape=(#sets, #vectors, 3),
        returns shape=(#sets) and the structure factor is normalized by
        the number of vectors in each set and the number of frames
    """
    kw = dict(ns=ns, nt=nt, dt=dt)
    if q.ndim == 2:
        return si_vector_self_intermediate(q, tr, **kw)
    elif q.ndim == 3:
        return np.asarray([np.mean(si_vector_self_intermediate(v, tr, **kw))
                           for v in progress_bar(q)])


@namedtuplefy
def si_self_intermediate_spherical(q_mod, tr, ns=200, nt=100, dt=1):
    r"""
    Single atom self incoherent structure factor for a set of q-vector moduli,
    spherically averaged in q-vector space

    The analytical spherical average is carried out.

    Parameters
    ----------
    q_mod: numpy.ndarray
        Array of q-vector moduli
    tr: numpy.ndarray
        atomic trajectory for one atom, shape=(#frames, 3)
    ns: int
        Number of t0's for a given time lapse `t`, i.e., number of
        sampling starting times.
    nt: int
        Number of t's, i.e, number time lapses
    dt: int
        Spacing between consecutive time lapses, in units of number of frames

    Returns
    -------
    namedtuple
        Fields of the namedtuple
        sf: numpy.ndarray, structure factor of shape (q_mod.shape, nt)
        t: numpy.ndarray, list of time lapses where the structure factor
            is calculated
    """
    co = 1e-05  # avoid later dividing by zero
    n_q = len(q_mod)
    nfr = len(tr)

    it = 0  # current number of evaluated time lapses
    t = 0  # first time lapse is no lapse
    sf = np.zeros((nt, n_q))  # shape is (nt, n_q)
    ts = []
    while it < nt:  # cycle over all t's (+1.0 for t=0)
        dt0 = max(1, int((nfr - t) / ns))  # separation between t0's
        t0s = np.arange(1, nfr - t, dt0)  # from 1 to nfr-t every delta
        dij = np.linalg.norm(tr[t0s] - tr[t0s + t], axis=1)  # distances
        phase = co + np.outer(q_mod, dij)
        sf[it] = (np.sin(phase) / phase).mean(axis=1)
        ts.append(t)
        it += 1
        t += dt
    return dict(sf=sf.transpose(), t=np.asarray(ts))


@namedtuplefy
def self_intermediate_spherical(q_mod, tr, bi, ns=200, nt=100, dt=1,
                                n_cores=None):
    r"""
    Self incoherent structure factor for a set of q-vector moduli,
    spherically averaged in q-vector space and by the sum of the
    squares of the scattering lengths

    The analytical spherical average is carried out.

    Parameters
    ----------
    q_mod: numpy.ndarray
        Array of q-vector moduli
    tr: numpy.ndarray
        atomic trajectory for one atom, shape=(#atoms, #frames, 3)
    bi: numpy.ndarray
        incoherent scattering lengths
    ns: int
        Number of t0's for a given time lapse `t`, i.e., number of
        sampling starting times.
    nt: int
        Number of t's, i.e, number time lapses
    dt: int
        Spacing between consecutive time lapses, in units of number of frames
    n_cores: int
        Number of CPU cores to use. `None` means use all cores

    Returns
    -------
    namedtuple
        Fields of the namedtuple
        sf: numpy.ndarray, structure factor of shape (q_mod.shape, nt)
        t: numpy.ndarray, list of time lapses where the structure factor
            is calculated
    """
    sf = np.zeros((len(q_mod), nt))

    def serial_worker(atomic_tr):
        return si_self_intermediate_spherical(q_mod, atomic_tr, ns=ns, nt=nt,
                                              dt=dt).sf
    if n_cores == 1:
        for i, atomic_tr in enumerate(tr):
            sf += bi[i] ** 2 * serial_worker(atomic_tr)
    else:
        if n_cores is None:
            n_cores = multiprocessing.cpu_count() - 1
        pool = pathos.pools.ProcessPool(ncpus=n_cores)
        for i, _sf in enumerate(pool.map(serial_worker, tr)):
            sf += bi[i] ** 2 * _sf
        pool.terminate()
    return dict(sf=sf / sum(bi * bi), t=dt * np.arange(nt))
