from __future__ import (print_function, absolute_import)

import pytest
from pytest import approx
import numpy as np

from nscsim.quasielastic import incoherent


def diffusive_particles(n_atoms, jump, duration):
    r"""
    Trajectory coordinates for particles undergoing diffusion

    Expected :math:`I(Q,t) = e^{-0.5 \cdot Q^2 \cdot t}`

    Parameters
    ----------
    n_atoms: int
        Number of particles
    jump: float
        Elemental displacement
    duration: int
        Number of time steps

    Returns
    -------
    numpy.ndarray
        Array of shape (n_atoms, duration, 3)
    """

    tr = np.random.choice([-1, 1], 3 * n_atoms * duration)
    tr = tr.reshape(n_atoms, 3, duration)
    tr = np.cumsum(tr, axis=2)
    return jump * np.transpose(tr, (0, 2, 1))


def test_si_self_intermediate_spherical():
    q = np.array((0.2, 0.3, 0.4, 0.5))
    tr = diffusive_particles(1, 1.0, 10000)[0]
    out = incoherent.si_self_intermediate_spherical(q, tr, ns=5000, nt=100)
    time_at_FWHM = np.abs(out.sf - 0.5).argmin(axis=1)
    assert np.log(2)/(time_at_FWHM * np.square(q)) == approx(0.5, abs=0.1)


def test_self_intermediate_spherical():
    q = (0.2, 0.3, 0.4, 0.5)
    n_atoms = 100
    tr = diffusive_particles(n_atoms, 1.0, 10000)
    bi = np.ones(n_atoms)
    out = incoherent.self_intermediate_spherical(q, tr, bi, ns=1000, nt=100)
    time_at_FWHM = np.abs(out.sf - 0.5).argmin(axis=1)
    assert np.log(2)/(time_at_FWHM * np.square(q)) == approx(0.5, abs=0.1)


if __name__ == '__main__':
    pytest.main()
