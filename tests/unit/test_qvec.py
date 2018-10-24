from __future__ import (print_function, absolute_import)

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from math import isclose, sqrt
import pytest

from nscsim import qvec


def test_unit_sphere_surf():
    v = qvec.unit_sphere_surf(42)
    assert len(v) == 42
    assert sum(np.linalg.norm(v, axis=1)) == 42


def test_moduli_linscale():
    for qs in (qvec.moduli_linscale(0, q_mod_max=1.0, q_mod_delta=0.1),
               qvec.moduli_linscale(0, q_mod_max=1.0, n_q_mod=10),
               qvec.moduli_linscale(0, q_mod_delta=0.1, n_q_mod=10)):
        assert np.array_equal(qs, np.arange(0, 1.0, 0.1))


def test_moduli_logscale():
    qs = qvec.moduli_logscale(-1, 2)
    assert np.array_equal(qs, np.logspace(-1, 2, num=30))


def test_sphere_average():
    qs = qvec.sphere_average(range(1, 10))
    assert np.array_equal(9 * qs[0], qs[-1])


def test_reciprocal_descriptors():
    def d2l(d):
        return [d['a_r'], d['b_r'], d['c_r'],
                d['alpha_r'], d['beta_r'], d['gamma_r']]
    # Orthorhombic lattice
    d = qvec.reciprocal_descriptors([1., 2., 3., 90., 90., 90.])
    assert_almost_equal(d2l(d), [1., 1./2, 1./3, 90., 90., 90.], decimal=2)
    np.array_equal(d['m_b'], np.identity(3))
    # fcc standard reduced cell (rhombohedron)
    f = sqrt(3./2)
    d = qvec.reciprocal_descriptors([1., 1., 1., 60., 60., 60.])
    assert_almost_equal(d2l(d),
                        [f, f, f, 109.47, 109.47, 109.47], decimal=2)
    # bcc standard reduced cell
    d = qvec.reciprocal_descriptors([1., 1., 1., 109.47, 109.47, 109.47])
    assert_almost_equal(d2l(d), [f, f, f, 60., 60., 60.], decimal=2)
    # hexagonal lattice
    d = qvec.reciprocal_descriptors([1., 1., 2., 90., 90., 120.])
    f = np.sin(np.deg2rad(120))
    assert_almost_equal(d2l(d), [1./f, 1./f, 1./2, 90., 90., 60.])
    # monoclinic
    d = qvec.reciprocal_descriptors([1., 1., 2., 90, 90, 45])
    assert_almost_equal(d2l(d), [1.41, 1.41, 1./2, 90., 90., 135.], decimal=2)


def test_reciprocal_max_indexes():
    abc_r = (1.0, 2.0, 3.0)
    assert_array_equal(qvec.reciprocal_max_indexes(13.0, abc_r), (3, 2, 1))


def test_reciprocal_qvectors():
    d = qvec.reciprocal_descriptors([2*np.pi, 2*np.pi, 2*np.pi, 90., 90., 90])
    qv = qvec.reciprocal_qvectors(1, 1, 1, m_b=d['m_b'])
    qv = qv.reshape(3, 27).transpose()
    assert_almost_equal(qv[0], (-1, -1, -1), decimal=3)
    assert_almost_equal(qv[-1], (1, 1, 1), decimal=3)


def test_reciprocal_qmoduli():
    d = qvec.reciprocal_descriptors([2*np.pi, 2*np.pi, 2*np.pi, 90., 90., 90])
    rm = qvec.reciprocal_qmoduli(1, 1, 0, m_b=d['m_b']).flatten()
    f=np.sqrt(2)
    assert_almost_equal(rm, (f, 1, f, 1, 0, 1, f, 1, f), decimal=3)


def test_reciprocal_average():
    lt = (2*np.pi, 2*np.pi, 2*np.pi, 90, 90, 90)
    q_mod_bins = (0, 1.01, np.sqrt(2)+0.01, np.sqrt(3)+0.01)
    q_arrays = qvec.reciprocal_average(q_mod_bins, lt)
    assert_array_equal([len(s) for s in q_arrays], [7, 12, 8])


if __name__ == '__main__':
    pytest.main()
