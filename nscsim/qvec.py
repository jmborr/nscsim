from __future__ import (print_function, absolute_import)

import numpy as np
from scipy.linalg import norm
from math import pi, cos, sin, sqrt, acos


def unit_sphere_surf(nvec=200):
    r"""
    Vectors randomly sampling the surface of the unit sphere.

    Ensures that for every vector v in the sampling, inverse vector -v is
    also included.

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
    v = np.random.normal(size=(int(nvec/2), 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return np.append(v, -v, axis=0)


def moduli_linscale(q_mod_min, q_mod_max=None, q_mod_delta=None,
                    n_q_mod=None):
    r"""
    An array of q vector moduli (q_mods) is generated depending on the passed
    arguments. These are the options:
    1. (q_mod_min, q_mod_max, q_mod_delta)
    2. (q_mod_min, q_mod_max, n_q_mod) with guaranteed q_mod_min and q_mod_max
    3. (q_mod_min, q_mod_delta, n_q_mod)

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
            q_mods = np.linspace(q_mod_min, q_mod_max, num=n_q_mod, endpoint=True)
    elif q_mod_delta is not None and n_q_mod is not None:
        # q_mods from triad (q_mod_min, q_mod_delta, n_q_mod)
        q_mods = q_mod_min + q_mod_delta * np.arange(n_q_mod)

    if q_mods is None:
        raise ValueError('Insufficient arguments passed.'
                         ' No q-vector moduli generated')
    return q_mods


def moduli_logscale(min_exp, max_exp, n_per_base=10, base=10):
    r"""
    An array of q vector moduli (q_mods) on a logarithmic scale.

    Parameters
    ----------
    min_exp: float
        minimum q modulus is base**min_exp. Example: 0.1 = 10**(-1)
    max_exp: float
        maximum q modulus is base**max_exp. Exaple: 100 = 10**2
    n_per_base: int
        number of q moduli between base**N and base**(N+1)
    base: int
        base of the logarithm

    Returns
    -------
    numpy.ndarray
        Array of q-vector moduli
    """
    n_mods = (max_exp - min_exp) * n_per_base
    return np.logspace(min_exp, max_exp, num=n_mods, endpoint=True, base=base)


def sphere_average(q_mod_array, nvec=200):
    r"""
    Set of q-vectors with random orientations and of specified moduli.

    Parameters
    ----------
    q_mod_array:
        Array of q-vector moduli (list, tuple, numpy.ndarray, iterator..)
    nvec: int
        Number of orientations. Must be an even number
    Returns
    -------
    numpy.ndarray
        Array of vectors, shape = (len(q_mods), nvec, 3)
    """

    q_mods = np.asarray(q_mod_array)
    return np.tensordot(q_mods, unit_sphere_surf(nvec), axes=0)


def reciprocal_descriptors(lattice):
    r"""
    Compute parameters of interest of the reciprocal lattice.

    Crystallographic convention of defining the reciprocal lattice vectors
    with no $2\pi$ prefactor.

    See http://docs.mantidproject.org/nightly/concepts/Lattice.html as well as
    https://github.com/mantidproject/documents/blob/master/Design/UBMatriximplementationnotes.pdf
    for details of the calculations

    Parameters
    ----------
    lattice: list
        List of lattice sizes and angles (a, b, c, alpha, beta, gamma). Angle
        units in degrees.

    Returns
    -------
    dict
        Dictionary of reciprocal lattice parameters and derived quantities
        a_r, b_r, c_r: lattice sizes
        alpha_r, beta_r, gamma_r: angles, in degrees
        m_b: matrix where columns are the components of the reciprocal
            lattice vectors, assuming a oriented lattice.
        f_vol: Volume of the direct cell in fractional coordinates
    """
    # Collect lattice parameters
    a, b, c = lattice[0:3]
    d2r = pi / 180.  # from degrees to radians
    alpha, beta, gamma = [d2r * x for x in lattice[3:]]

    # Volume of the direct cell in fractional coordinates
    f_vol = sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 +
                 2 * cos(alpha) * cos(beta) * cos(gamma))

    # reciprocal lattice sides
    a_r = sin(alpha) / (f_vol * a)
    b_r = sin(beta) / (f_vol * b)
    c_r = sin(gamma) / (f_vol * c)

    # reciprocal lattice angles
    alpha_r = acos((cos(beta)*cos(gamma)-cos(alpha)) / (sin(beta)*sin(gamma)))
    beta_r = acos((cos(gamma)*cos(alpha)-cos(beta)) / (sin(gamma)*sin(alpha)))
    gamma_r = acos((cos(alpha)*cos(beta)-cos(gamma)) / (sin(alpha)*sin(beta)))

    # Components of the reciprocal vectors {a*, b*, c*} in an orthogonal
    # system with units vectors {i*, j*, k*}
    # We enforce the oriented lattice convention
    a_vr = [a_r, 0., 0.]  # vector a* along direction of unit vector i*
    b_vr = [b_r*cos(gamma_r), b_r*sin(gamma_r), 0.]  # b* on the {i*,j*} plane
    c_vr = [c_r*cos(beta_r), -c_r*sin(beta_r)*cos(alpha), 1./c]

    # Matrix of reciprocal unit vectors, with components as columns. This
    # matrix transform the coordinates from the system of reference spanned by
    # the reciprocal lattice vectors {a*, b*, c*} to the system of reference
    # spanned by the orthogonal vectors {i*, j*, k*}
    # For any vector v = h a* + k b* + l c*, (h, k, l) miller indexes, then
    # numpy.matmul(m_b, (h, k, l)) are the components of v in the {i*, j*, k*}
    # system of reference
    m_b = np.asarray([a_vr, b_vr, c_vr]).transpose()

    return dict(a_r=a_r, b_r=b_r, c_r=c_r,
                alpha_r=alpha_r/d2r, beta_r=beta_r/d2r, gamma_r=gamma_r/d2r,
                m_b=m_b, f_vol=f_vol)


def reciprocal_max_indexes(q_mod, abc_r):
    r"""
    Positive h, k, l indexes encompassing q_mod.

    Consider a vector $\vec{Q}$ with modulus $q_{mod}$ pointing in the
    direction of reciprocal vector $\vec{a^*}$. Then,
    $q_{mod} = 2 \pi h |\vec{a^*}|$ with $h$ the first Miller index.
    Similarly for $\vec{Q}$ pointing along $\vec{b^*}$ and $\vec{c^*}$

    Parameters
    ----------
    q_mod: float
        Momentum transfer modulus
    abc_r: numpy.ndarray
        The three sizes of the reciprocal lattice

    Returns
    -------
    numpy.ndarray
        Maximum Miller indexes
    """
    np_abc_r = np.asarray(abc_r)  # coerce to numpy array, just in case
    return 1 + (q_mod / (2 * np.pi * np_abc_r)).astype(int)


def reciprocal_qvectors(hm, km, lm, m_b=None):
    r"""
    Momentum transfer vectors for a grid of Miller indexes.

    The extent of the grid is [-h, h] x [-k, k] x [-l, l]. Q-vectors are
    represented by their projections on an orthonormal frame of reference.

    Parameters
    ----------
    hm: int
        Maximum value of the first Miller index
    km: int
        Maximum value of the second Miller index
    lm: int
        Maximum value of the third Miller index
    m_b: numpy.ndarray
        Matrix where columns are the components of the reciprocal lattice
        vectors in an orthonormal frame of reference following the
        crystallographic convention of an oriented lattice construction. If
        None, we assume the identity matrix.

    Returns
    -------
    numpy.ndarray
        shape = (3, 1+2*hm, 1+2*lm, 1+2*lm)
    """
    running_indexes = np.mgrid[-hm:1+hm, -km:1+km, -lm:1+lm]
    return 2 * np.pi * np.tensordot(m_b, running_indexes, 1)


def reciprocal_qmoduli(hm, km, lm, m_b=None):
    r"""
    Momentum transfer moduli for a grid of Miller indexes.

    The extent of the grid is [-h, h] x [-k, k] x [-l, l].

    Parameters
    ----------
    hm: int
        Maximum value of the first Miller index
    km: int
        Maximum value of the second Miller index
    lm: int
        Maximum value of the third Miller index
    m_b: numpy.ndarray
        Matrix where columns are the components of the reciprocal lattice
        vectors in an orthonormal frame of reference following the
        crystallographic convention of an oriented lattice construction. If
        None, we assume the identity matrix.

    Returns
    -------
    numpy.ndarray
        shape = (1+2*hm, 1+2*lm, 1+2*lm)
    """
    return norm(reciprocal_qvectors(hm, km, lm, m_b=m_b), axis=0)


def reciprocal_average(q_mod_bins, lattice, max_nvec=200):
    r"""
    List of arrays of q-vectors, each array containing q-vectors of moduli
    within two consecutive bins of `q_mod_bins`

    Parameters
    ----------
    q_mod_bins: numpy.ndarray
        Sorted array of Q moduli representing bin boundaries.
    lattice: numpy.ndarray
        Direct lattice parameters (a, b, c, alpha, beta, gamma)
    max_nvec: int
        Maximum number of Q vectors to be retained within each bin. Must be
        even number

    Returns
    -------
    q_arrays: list
        list items are arrays of Q-vectors with len(qvecs) = len(q_mod_bins)-1.
        qvecs[i] is array of shape (max_nvec, 3) containing Q-vectors with
        module in between q_mod_bins[i] and q_mod_bins[i+1]
    """
    if max_nvec % 2 != 0:
        raise ValueError('max_nvec needs to be an even number')

    # Find momentum transfer vector and moduli of a reciprocal grid
    # in fractional coordinates
    r = reciprocal_descriptors(lattice)
    q_mod_max = np.max(q_mod_bins)
    abc_r = (r['a_r'], r['b_r'], r['c_r'])  # lattice sizes
    hm, km, lm = reciprocal_max_indexes(q_mod_max, abc_r)
    moduli = reciprocal_qmoduli(hm, km, lm, r['m_b']).flatten()
    qvecs = reciprocal_qvectors(hm, km, lm, r['m_b'])
    qvecs = qvecs.reshape(3, len(moduli)).transpose()

    q_arrays = list()
    for i in range(len(q_mod_bins) - 1):
        low_q = q_mod_bins[i]
        high_q = q_mod_bins[i+1]
        indexes = np.where((moduli >= low_q) & (moduli < high_q))
        selected = qvecs[indexes]
        if len(selected) > max_nvec:
            np.random.shuffle(selected)
            selected = selected[0: int(max_nvec/2)]  # shape (max_nvec/2, 3)
            # For every selected Q-vector, collect its reflection
            q_arrays.append(np.concatenate((selected, -selected), axis=0))
        else:
            q_arrays.append(selected)
    return q_arrays
