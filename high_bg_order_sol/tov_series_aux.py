# tov_series_aux.py
import numpy as np

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _x(K: float, pc: float) -> float:
    """
    x = sqrt(K*pc)
    """
    if K <= 0.0 or pc <= 0.0:
        raise ValueError("Require K>0 and pc>0 for this n=1 polytrope EOS series.")
    return np.sqrt(K * pc)

def _asarray(r):
    return np.asarray(r, dtype=float)


# ------------------------------------------------------------
# Series approximations near r=0 (up to r^8)
# EOS: rho = p + sqrt(p/K), with central pc
# These come from your Mathematica outputs for:
#   rhoSeries, mSeries, nuPoly, lambdaPoly
#
# Conventions:
#   m(r) is the mass-aspect function used in TOV:
#     e^{-lambda} = 1 - 2m/r
#   nu(r) has an additive constant nu_c (gauge)
# ------------------------------------------------------------

def rho_approx(r, K: float, pc: float):
    """
    rho(r) series up to O(r^8) from your Out[597] (rhoSeries).
    """
    r = _asarray(r)
    x = _x(K, pc)  # sqrt(K pc)

    # Out[597]:
    # rho = pc + pc/x
    #       - ((2Kpc + x)(4Kpc + x)(1+2x) pi r^2)/(3 K^2 x)
    #       + ((1+4Kpc+4x)(34Kpc + 448K^2 pc^2 + 3x + 200 (Kpc)^(3/2)) pi^2 r^4)/(90 K^3)
    #       - ((2Kpc+x)(3+580Kpc+...+62080 (Kpc)^(5/2)) pi^3 r^6)/(1890 K^4)
    #       + ( ... ) pi^4 r^8 /(340200 K^5)
    #
    Kpc = K * pc

    term0 = pc + pc / x

    term2 = -((2.0*Kpc + x) * (4.0*Kpc + x) * (1.0 + 2.0*x) * np.pi * r**2) / (3.0 * K**2 * x)

    term4_num = (1.0 + 4.0*Kpc + 4.0*x) * (34.0*Kpc + 448.0*K**2 * pc**2 + 3.0*x + 200.0*(Kpc**1.5))
    term4 = (term4_num * (np.pi**2) * r**4) / (90.0 * K**3)

    term6_poly = (
        3.0
        + 580.0*Kpc
        + 25632.0*(Kpc**2)
        + 57344.0*(Kpc**3)
        + 24.0*x
        + 5552.0*(Kpc**1.5)
        + 62080.0*(Kpc**2.5)
    )
    term6 = -((2.0*Kpc + x) * term6_poly * (np.pi**3) * r**6) / (1890.0 * K**4)

    term8_poly = (
        54591488.0*(K**5)*(pc**5)
        + 15.0*x
        - 2.0*Kpc*(669.0 + 3050.0*x)
        + 8704.0*(K**4)*(pc**4)*(7215.0 + 10492.0*x)
        + 384.0*(K**3)*(pc**3)*(13745.0 + 61476.0*x)
        + 8.0*(K**2)*(pc**2)*(4313.0 + 83508.0*x)
    )
    term8 = (term8_poly * (np.pi**4) * r**8) / (340200.0 * K**5)

    return term0 + term2 + term4 + term6 + term8


def m_approx(r, K: float, pc: float):
    """
    m(r) series up to O(r^9) from your Out[598] (mSeries).
    Note: your mSeries is already the full m(r) (not divided by 4pi r^3).
    """
    r = _asarray(r)
    x = _x(K, pc)
    Kpc = K * pc

    # Out[598]:
    # m = (2*pi*r^3)/(8505 K^4) * [ 5670 K^3 (Kpc + x)
    #     - (1134 K^2 (2Kpc+x)(4Kpc+x)(1+2x) pi r^2)/x
    #     + 27 K (1+4Kpc+4x)(34Kpc+448K^2 pc^2+3x+200(Kpc)^(3/2)) pi^2 r^4
    #     - (2Kpc+x)(...) pi^3 r^6
    #   ]
    #
    # where "(...)" in last term matches the same poly used in rho's r^6 coefficient.
    poly6 = (
        3.0
        + 580.0*Kpc
        + 25632.0*(Kpc**2)
        + 57344.0*(Kpc**3)
        + 24.0*x
        + 5552.0*(Kpc**1.5)
        + 62080.0*(Kpc**2.5)
    )

    A0 = 5670.0 * (K**3) * (Kpc + x)

    A2 = - (1134.0 * (K**2) * (2.0*Kpc + x) * (4.0*Kpc + x) * (1.0 + 2.0*x) * np.pi * r**2) / x

    A4_num = (1.0 + 4.0*Kpc + 4.0*x) * (34.0*Kpc + 448.0*(K**2)*(pc**2) + 3.0*x + 200.0*(Kpc**1.5))
    A4 = 27.0 * K * A4_num * (np.pi**2) * r**4

    A6 = - (2.0*Kpc + x) * poly6 * (np.pi**3) * r**6

    pref = (2.0 * np.pi * r**3) / (8505.0 * K**4)
    return pref * (A0 + A2 + A4 + A6)


def nu_approx(r, K: float, pc: float, nu_c: float = 0.0):
    """
    nu(r) series up to O(r^8) from your Out[655] (nuPoly).
    """
    r = _asarray(r)
    x = _x(K, pc)
    Kpc = K * pc

    term2 = (4.0 * (4.0*Kpc + x) * np.pi * r**2) / (3.0 * K)

    term4_num = 2.0 * (pc**2) * (3.0 + 14.0*x + 8.0*Kpc*(5.0 + 16.0*x))
    term4 = - (term4_num * (np.pi**2) * r**4) / (45.0 * (Kpc**1.5))

    term6_num = 2.0 * x * (
        9.0 - 198.0*x
        + 64.0*(K**2)*(pc**2)*(-15.0 + 112.0*x)
        - 8.0*Kpc*(139.0 + 174.0*x)
    )
    term6 = (term6_num * (np.pi**3) * r**6) / (2835.0 * K**3)

    term8_num = x * (
        15.0 - 2874.0*x
        + 512.0*(K**3)*(pc**3)*(-785.0 + 448.0*x)
        - 128.0*(K**2)*(pc**2)*(326.0 + 1103.0*x)
        + 8.0*Kpc*(-593.0 + 2398.0*x)
    )
    term8 = - (term8_num * (np.pi**4) * r**8) / (85050.0 * K**4)

    return nu_c + term2 + term4 + term6 + term8


def lambda_approx(r, K: float, pc: float):
    """
    lambda(r) series up to O(r^8) from your Out[660] (lambdaPoly).
    """
    r = _asarray(r)
    x = _x(K, pc)
    Kpc = K * pc

    # Out[660]:
    # lambda = (4*pi*r^2)/(42525 K^4) * [
    #   28350 K^3 (Kpc + x)
    #   - 1890 K^2 x (3 + 4x + 4Kpc(5+7x)) pi r^2
    #   + 15 K x (27 - 594x + 160 K^2 pc^2 (3+28x) - 4Kpc(365+582x)) pi^2 r^4
    #   - x (15 - 2442x + 256 K^3 pc^3 (-310+203x) + 4Kpc(407+818x) - 32 K^2 pc^2 (557+2576x)) pi^3 r^6
    # ]
    B0 = 28350.0 * (K**3) * (Kpc + x)

    B2 = - 1890.0 * (K**2) * x * (3.0 + 4.0*x + 4.0*Kpc*(5.0 + 7.0*x)) * np.pi * r**2

    B4 = 15.0 * K * x * (
        27.0 - 594.0*x
        + 160.0*(K**2)*(pc**2)*(3.0 + 28.0*x)
        - 4.0*Kpc*(365.0 + 582.0*x)
    ) * (np.pi**2) * r**4

    B6 = - x * (
        15.0 - 2442.0*x
        + 256.0*(K**3)*(pc**3)*(-310.0 + 203.0*x)
        + 4.0*Kpc*(407.0 + 818.0*x)
        - 32.0*(K**2)*(pc**2)*(557.0 + 2576.0*x)
    ) * (np.pi**3) * r**6

    pref = (4.0 * np.pi * r**2) / (42525.0 * K**4)
    return pref * (B0 + B2 + B4 + B6)


def p_approx(r, K: float, pc: float):
    """
    p(r) series up to O(r^8).

    You *can* derive pSeries from your p2/p4/p6/p8, but your Out[596]
    is already a closed-form polynomial. To avoid transcription errors,
    compute p(r) by inverting EOS at the series rho(r) if you want,
    but easiest is to use your direct pSeries expression.

    Here I implement pSeries based on the *known identity* rho = p + sqrt(p/K)
    by solving for p in terms of rho:
      Let s = sqrt(p/K) => p = K s^2, rho = K s^2 + s
      => K s^2 + s - rho = 0 => s = (-1 + sqrt(1 + 4K rho))/(2K)
      => p = K s^2
    Using rho_approx(r) gives p_approx(r) consistent with your EOS.
    """
    r = _asarray(r)
    rho = rho_approx(r, K=K, pc=pc)

    disc = 1.0 + 4.0*K*rho
    s = (-1.0 + np.sqrt(disc)) / (2.0*K)
    p = K * s**2
    # numerical guard
    return np.maximum(p, 0.0)
