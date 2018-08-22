from __future__ import (print_function, absolute_import)

import numpy as np
from tqdm import tqdm as progress_bar


def vector_structure_factor(q, tr, b_c):
    """
    Calculate the static structure factor for each vector of a set of q-vectors

    Parameters
    ----------
    q: numpy.ndarray
        q vectors shape=(#vectors, 3)
    tr: numpy.ndarray
        atomic trajectory shape=(#frames, #atoms, 3)
    b_c: numpy.ndarray
        Real part of the bound coherent scattering lengths shape=(#atoms)

    Returns
    -------
    numpy.ndarray
        array of shape=(#vectors). Structure factor normalized by number
        of frames, number of atoms, and the square of the average scattering
        length
    """
    sq = list()
    # iterate over each q-vector, v.shape=(3,)
    for v in q:
        s = 0.0
        # iterate over each frame, fr.shape = (atoms, 3)
        for fr in tr:
            vfr = np.dot(fr, v)  # vfr.shape=(#atoms)
            real = np.dot(b_c, np.cos(vfr))
            imag = np.dot(b_c, np.sin(vfr))
            s += (real * real) + (imag * imag)
        sq.append(s)
    return np.asarray(sq) / (len(tr) * np.square(np.sum(b_c)))


def structure_factor(q, tr, b_c):
    """
    Calculate the static structure factor for each vector of a set of q-vectors
    or for each set average in a list of sets of q-vectors.

    Parameters
    ----------
    q: numpy.ndarray
        q vectors shape=(#vectors, 3) for a set of q-vectors or
        shape=(#sets, #vectors, 3) for a list of sets of q-vectors
    tr: numpy.ndarray
        atomic trajectory shape=(#frames, #atoms, 3)
    b_c: numpy.ndarray
        Real part of the bound coherent scattering lengths shape=(#atoms)

    Returns
    -------
    numpy.ndarray
        If q.shape=(#vectors, 3), returns shape=(#vectors) and the structure
        factor is normalized by the number of frames, number of atoms,
        and the square of the average scattering length. If
        q.shape=(#sets, #vectors, 3), returns shape=(#sets) and the
        structure factor is normalized by the number of vectors in each set,
        the number of frames, the number of atoms, and the square of
        the average scattering length.
    """
    if q.ndim == 2:
        return vector_structure_factor(q, tr, b_c)
    elif q.ndim == 3:
        return np.asarray([np.mean(vector_structure_factor(v, tr, b_c))
                           for v in progress_bar(q)])
