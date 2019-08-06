from __future__ import (print_function, absolute_import)

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from nscsim.utilities import shared_array

from nscsim.quasielastic import incoherent2


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
    tr = np.random.choice([-1, 1], (n_atoms, 3, duration))
    tr = np.cumsum(tr, axis=2)
    tr = jump * np.transpose(tr, (0, 2, 1))
    return shared_array(from_array=tr)


def test_intermediate_spherical():
    q = np.array((0.2, 0.3, 0.4, 0.5))
    n_atoms = 100
    tr = diffusive_particles(n_atoms, 1.0, 1000)
    bi = np.ones(n_atoms)
    serial = incoherent2.intermediate_spherical(tr, q, bi, n_cores=1)
    parall = incoherent2.intermediate_spherical(tr, q, bi, n_cores=2)
    assert_almost_equal(serial, parall, decimal=2)
    time_at_fwhm = np.abs(parall - 0.5).argmin(axis=1)
    assert_almost_equal(np.log(2)/(time_at_fwhm * np.square(q)),
                        0.5 * np.ones(len(q)), decimal=1)


if __name__ == '__main__':
    pytest.main()
