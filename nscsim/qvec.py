from __future__ import (print_function, absolute_import)

import numpy as np
from math import pi, cos, sin, sqrt, acos, asin


def unit_sphere_surf(nvec=200):
    """
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
    v = np.random.normal(size=(nvec/2, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return np.append(v, -v, axis=0)


def moduli_linscale(q_mod_min, q_mod_max=None, q_mod_delta=None,
                    n_q_mod=None):
    """
    An array of q vector moduli (q_mods) is generated depending on the passed
    arguments. These are the options:
    1. (q_mod_min, q_mod_max, q_mod_delta) generates q_mods
    2. (q_mod_min, q_mod_max, n_q_mod) generates q_mods
    3. (q_mod_min, q_mod_delta, n_q_mod) generates q_mods

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
            # q_mods from triad (q_mod_min, q_mod_max, n_q_mod)
            q_mod_delta = (q_mod_max - q_mod_min) / n_q_mod
            q_mods = np.arange(q_mod_min, q_mod_max, q_mod_delta)
    elif q_mod_delta is not None and n_q_mod is not None:
        # q_mods from triad (q_mod_min, q_mod_delta, n_q_mod)
        q_mods = q_mod_min + q_mod_delta * np.arange(n_q_mod)

    if q_mods is None:
        raise ValueError('Insufficient arguments passed.'
                         ' No q-vector moduli generated')
    return q_mods


def moduli_logscale(min_exp, max_exp, n_per_base=10, base=10):
    """
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
    """
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
    """
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
    """

    # Collect lattice parameters
    a, b, c = lattice[0:3]
    d2r = pi / 180.  # from degrees to radians
    alpha, beta, gamma = d2r * lattice[3:]

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
                m_b=m_b)


def reciprocal_max_indexes(q_mod, abc_r):
    """
    Positive h, k, l indexes encompassing q_mod.

    Consider a vector $\vec{Q}$ with modulus $q_{mod}$ pointing in the direction
    of reciprocal vector $\vec{a^*}$. Then, $q_{mod} = 2 \pi h |\vec{a^*}|$
    with $h$ the first Miller index. Similarly for $\vec{Q}$ pointing along
    $\vec{b^*}$ and $\vec{c^*}$

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
    return 1 + int(q_mod / (2 * np.pi * abc_r))


def reciprocal_moduli(hm, km, lm, m_b=None):
    """
    Momentum transfer moduli of a grid of Miller indexes

    The extent of the grid is [-h, h] x [-k, k] x [-l, l]

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
        vectors in an orthogonal frame of reference following the
        crystallographic convention of an oriented lattice construction. If
        None, we assume the identity matrix.

    Returns
    -------
    numpy.ndarray

    """
    na, nb, nc = hm, km, lm
    moduli = np.zeros((2*na+1, 2*nb+1, 2*nc+1))
    # TBD this could be achieved with numpy arrays
    for h in range(-na, 1+na):
        for k in range(-nb, 1+nb):
            for l in range(-nc, 1+nc):
                moduli[h][k][l] = sum(np.square(np.dot(m_b, np.asarray(h, k, l))))
    return 2 * pi * np.sqrt(moduli)


def reciprocal_average(q_mod_array, lattice, min_nvec=1, max_nvec=200):
    """
    Set of reciprocal lattice vectors with moduli close to q_mod_array.

    For q_mod_array[i], find the reciprocal lattice vectors in the range
    ( (q_mod_array[i-1] + q_mod_array[i])/2,
      (q_mod_array[i] + q_mod_array[i+1])/2 )
    and retain only nvec vectors (randomly selected)

    There may be that there are few or no reciprocal lattice vectors for some
    q_mod_array[i], specially for low values of the momentum transfer modulus.
    Thus, we return only those moduli of q_mod_array for which sufficient
    reciprocal lattice vectors are found

    Parameters
    ----------
    q_mod_array
    lattice
    min_nvec
    max_nvec

    Returns
    -------
    (q_mods, q_vecs): tuple
        q_mods: moduli of q_mod_array for which reciprocal lattice vectors are
            found.
        q_vecs: reciprocal lattice vectors for each modulus of q_mods. Thus,
            len(q_vecs) is len(q_mods)
    """
    r = reciprocal_descriptors(lattice)

    # Find momentum transfer moduli for a reciprocal grid in fractional coords
    q_mod_max = np.max(q_mod_array)
    abc_r = np.asarray(r['a_r'], r['b_r'], r['c_r'])  # lattice sizes
    hkl_max = reciprocal_max_indexes(q_mod_max, abc_r)
    hkl_moduli = reciprocal_moduli(hkl_max, r['m_b'])

    # create lower and upper bin boundaries for the array of momentum transfers
    i_bins = (q_mod_array[:-1] + q_mod_array[1:]) / 2.  # inner bins
    l_bins = np.insert(i_bins, 0, 2 * q_mod_array[0] - i_bins[0])  # lower
    u_bins = np.append(i_bins, 2 * q_mod_array[-1] - i_bins[-1])  # upper

    # Find momentum transfer vectors within each q_mod bin
    q_mods = list()
    q_vecs = list()
    for (lower, upper) in zip(l_bins, u_bins):
        hkl = np.argwhere((hkl_moduli > lower) & (hkl_moduli < upper))
        if len(hkl) < min_nvec:
            continue  # insufficient number of lattice vectors in this bin
        if len(hkl) > max_nvec:  # excessive number of lattice vectors
            hkl = np.random.shuffle(hkl)[0: max_nvec]
        q_mods.append((lower + upper) / 2)
        q_vecs.append(hkl)

    return np.asarry(q_mods), np.asarray(q_vecs)
