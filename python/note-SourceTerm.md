## Note on modulating the source term

A low-frequency expansion on the perturbation parameter $H$ takes the form:
$$
H = H^{(0)} + (\omega M)H^{(1)} +  (\omega M)^2 H^{(2)} + \mathcal{O}((\omega M)^3)
$$
Despite the coefficients of every orders taking different form, they all possesses a zeroth-order-like contribution, where such contribution is denoted with:
$$
\mathcal{L}^+_\ell,
$$
where $\ell$ is the angular order in the expansion of the perturbation. One can take the [Katagiri 2409](http://arxiv.org/abs/2409.18034v2) paper as reference. In this work, we have a clean zeroth order equation (this also defines the operator $\mathcal{L}$), given by the `L[h00_, w0_][r_]` function in the Mathematica notebook. Note we have eliminate the $w_0$ dependence from the homogeneous term.

To obtain the source term of the first order, we subtract `L[h01][r]` from the first order einstein equation for $H_0$. The remaining source term does not have $h_{01}$ contribution, or contribution from any of the higher order parameters. We can therefore obtain the full numerical behavior for $S_{h_0, 1}$ without numerically solving extra equations.

Using this method, the numerical behavior of $S_{h_0, 1}$ explodes at small $r$. This is likely due to some artifacts in the numerical procedure. We try to eliminate by estimating the center behavior with leading order contributions of the analytical solution.

We present the leading 2 orders of the analytical behavior of the perturbation parameters.

