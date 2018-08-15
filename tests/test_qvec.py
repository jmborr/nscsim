from __future__ import (print_function, absolute_import)

import numpy as np
import pytest

from nsc import qvec


def test_unit_sphere_surf():
    v = qvec.unit_sphere_surf(42)
    assert len(v) == 42
    assert sum(np.linalg.norm(v, axis=1)) == 42


def test_q_mod_array_generator():
    for qs in (qvec.q_mod_array_generator(0, q_mod_max=1.0, q_mod_delta=0.1),
               qvec.q_mod_array_generator(0, q_mod_max=1.0, n_q_mod=10),
               qvec.q_mod_array_generator(0, q_mod_delta=0.1, n_q_mod=10)):
        assert np.array_equal(qs, np.arange(0, 1.0, 0.1))


def test_sphere_average():
    for qs in (qvec.sphere_average(range(1, 10)),
               qvec.sphere_average(q_mod_min=1, q_mod_max=10, q_mod_delta=1),
               qvec.sphere_average(q_mod_min=1, q_mod_max=10, n_q_mod=9),
               qvec.sphere_average(q_mod_min=1, q_mod_delta=1, n_q_mod=9)):
        assert np.array_equal(9 * qs[0], qs[-1])


if __name__ == '__main__':
    pytest.main()
