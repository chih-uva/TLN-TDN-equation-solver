# h00_aux.py
import numpy as np


def _x(K: float, pc: float) -> float:
    """
    x = sqrt(K*pc)
    """
    if K <= 0.0 or pc <= 0.0:
        raise ValueError("Require K>0 and pc>0 for this n=1 polytrope EOS series.")
    return np.sqrt(K * pc)


def _asarray(r):
    return np.asarray(r, dtype=float)


def h00_approx(r, K: float, pc: float, a0: float, a8: float):
    """
    h00(r) series from the provided Mathematica expression.
    """
    r = _asarray(r)
    x = _x(K, pc)
    Kpc = K * pc

    term0 = a0

    term2 = -(
        a0
        * (3.0 + 26.0 * x + 8.0 * Kpc * (17.0 + 40.0 * x))
        * np.pi
        * r**2
    ) / (21.0 * (K + 4.0 * K * x))

    term4_num = (
        1687552.0 * (Kpc**4)
        + 15.0 * x
        + 2.0 * Kpc * (129.0 + 2480.0 * x)
        + 256.0 * (Kpc**3) * (4237.0 + 7844.0 * x)
        + 16.0 * (Kpc**2) * (3497.0 + 20572.0 * x)
    )
    term4_den = 1890.0 * (K**2) * (
        12.0 * Kpc + 64.0 * (Kpc**2) + x + 48.0 * (Kpc * x)
    )
    term4 = (a0 * term4_num * (np.pi**2) * r**4) / term4_den

    term6_num = (
        15.0
        - 318.0 * x
        + 104.0 * Kpc * (-7.0 + 2428.0 * x)
        + 2097152.0 * (Kpc**6) * (46867.0 + 20224.0 * x)
        + 122880.0 * (Kpc**4) * (80907.0 + 251804.0 * x)
        + 128.0 * (Kpc**2) * (41875.0 + 458607.0 * x)
        + 131072.0 * (Kpc**5) * (529047.0 + 805940.0 * x)
        + 1536.0 * (Kpc**3) * (282803.0 + 1553124.0 * x)
    )
    term6_den = 62370.0 * (K**3) * (
        1.0
        + 28.0 * x
        + 4096.0 * (Kpc**3) * (7.0 + 4.0 * x)
        + 1792.0 * (Kpc**2) * (5.0 + 12.0 * x)
        + 112.0 * Kpc * (3.0 + 20.0 * x)
    )
    term6 = -(a0 * term6_num * (np.pi**3) * r**6) / term6_den

    term8 = a8 * r**8

    return r**2 * (term0 + term2 + term4 + term6 + term8)


def h00p_approx(r, K: float, pc: float, a0: float, a8: float):
    """
    h00'(r) series from the provided Mathematica expression.
    """
    r = _asarray(r)
    x = _x(K, pc)
    Kpc = K * pc

    term1 = 2.0 * a0 * r

    term3 = -(
        4.0
        * a0
        * (3.0 + 26.0 * x + 8.0 * Kpc * (17.0 + 40.0 * x))
        * np.pi
        * r**3
    ) / (21.0 * (K + 4.0 * K * x))

    term5_num = (
        1687552.0 * (Kpc**4)
        + 15.0 * x
        + 2.0 * Kpc * (129.0 + 2480.0 * x)
        + 256.0 * (Kpc**3) * (4237.0 + 7844.0 * x)
        + 16.0 * (Kpc**2) * (3497.0 + 20572.0 * x)
    )
    term5_den = 315.0 * (K**2) * (
        12.0 * Kpc + 64.0 * (Kpc**2) + x + 48.0 * (Kpc * x)
    )
    term5 = (a0 * term5_num * (np.pi**2) * r**5) / term5_den

    term7_num = (
        15.0
        - 318.0 * x
        + 104.0 * Kpc * (-7.0 + 2428.0 * x)
        + 2097152.0 * (Kpc**6) * (46867.0 + 20224.0 * x)
        + 122880.0 * (Kpc**4) * (80907.0 + 251804.0 * x)
        + 128.0 * (Kpc**2) * (41875.0 + 458607.0 * x)
        + 131072.0 * (Kpc**5) * (529047.0 + 805940.0 * x)
        + 1536.0 * (Kpc**3) * (282803.0 + 1553124.0 * x)
    )
    term7_den = 31185.0 * (K**3) * (
        1.0
        + 28.0 * x
        + 4096.0 * (Kpc**3) * (7.0 + 4.0 * x)
        + 1792.0 * (Kpc**2) * (5.0 + 12.0 * x)
        + 112.0 * Kpc * (3.0 + 20.0 * x)
    )
    term7 = -(4.0 * a0 * term7_num * (np.pi**3) * r**7) / term7_den

    term9 = 10.0 * a8 * r**9

    return term1 + term3 + term5 + term7 + term9
