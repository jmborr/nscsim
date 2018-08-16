from __future__ import (print_function, absolute_import)

import numpy as np
import pytest

from nsc.elastic import structure_factor as sf


def test_vector_structure_factor():
    n_atoms = 5
    n_frames = 3
    n_vectors = 2
    shape = (n_frames, n_atoms, 3)
    tr = np.random.random_sample(n_frames * n_atoms * 3).reshape(shape)
    q = np.zeros(n_vectors * 3).reshape((n_vectors, 3))
    sl = np.ones(n_atoms)
    s = sf.vector_structure_factor(q, tr, sl)
    np.array_equal(s, np.ones(n_vectors))


def test_structure_factor():
    n_atoms = 5
    n_frames = 3
    n_sets = 7
    n_vectors = 2
    shape = (n_frames, n_atoms, 3)
    tr = np.random.random_sample(n_frames * n_atoms * 3).reshape(shape)
    q = np.zeros(n_sets * n_vectors * 3).reshape((n_sets, n_vectors, 3))
    sl = np.ones(n_atoms)
    s = sf.structure_factor(q, tr, sl)
    np.array_equal(s, np.ones(n_sets))


if __name__ == '__main__':
    pytest.main()
