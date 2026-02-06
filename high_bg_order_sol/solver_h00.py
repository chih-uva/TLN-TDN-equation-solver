# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import pickle
import sys
from pathlib import Path

# solution related imports
from EOS_aux import PolytropeEOS_n1
from solver_tov import cgs_to_geom, mass_g2Msun
from h00_aux import *


# expression of second derivative of h00
def h00pp_rhs(r, y, p_r, rho_r, lamb_r, drho, h=1e-4):
    """
    Compute the RHS of the h00 equation.
    
    Parameters:
        r: radial coordinate
        y: state vector [h00, h00']
        p_r: pressure at this point
        rho_r: energy density at this point
        lamb_r: lambda at this point (metric function)
        drho: derivative of energy density with respect to r
        h: step size (unused, kept for interface compatibility)
    
    Returns:
        [h00', h00''] derivative vector
    """
    
    def A(r):
        """Compute coefficient A(r) in the h00 equation"""
        lamb = lamb_r
        eL = np.exp(lamb)
        e2L = eL * eL
        e3L = e2L * eL
        pr = p_r
        rr = rho_r
        rrp = drho
        pi = np.pi
        r2 = r * r
        r3 = r * r * r
        r4 = r2 * r2
        r6 = r3 * r3
        
        term = (
            -1.0 - 3.0*eL + 3.0*e2L + e3L
            + 96.0*e2L*(-5.0 + 2.0*eL)*(pi**2)*r4*(pr**2)
            + 512.0*e3L*(pi**3)*r6*(pr**3)
            - 20.0*eL*(-1.0 + eL)*pi*r2*rr
            + 4.0*eL*pi*r2*pr*(15.0 - 9.0*eL + 6.0*e2L - 40.0*eL*pi*r2*rr)
            + 8.0*eL*pi*r3*rrp
        )
        return term
    
    def denom_D(r):
        """Compute denominator D(r)"""
        eL = np.exp(lamb_r)
        return r * r * (-1.0 + eL + 8.0*eL*np.pi*(r*r)*p_r)
    
    def hp_coeff(r):
        """Compute coefficient of h00'"""
        eL = np.exp(lamb_r)
        return (1.0 + eL + 4.0*eL*np.pi*(r*r)*p_r - 4.0*eL*np.pi*(r*r)*rho_r) / r
    
    def f(r, y):
        """RHS function for the ODE"""
        y1, y2 = y  # y1 = h00, y2 = h00'
        
        D = denom_D(r)
        if D == 0.0 or not np.isfinite(D):
            raise ZeroDivisionError(f"Denominator is singular at r={r}; adjust the integration domain or background solution.")
        
        y1_src = A(r) / D
        y2_src = -hp_coeff(r)
        
        # y' = [y2, y1_src*y1 + y2_src*y2]
        return np.array([y2, y1_src*y1 + y2_src*y2], dtype=float)
    
    return f(r, y)

# Solver for u = h00/r^2
def u_pp_rhs(r, Y, p_r, rho_r, lamb_r, drho):
    """
    RHS of the ODE satisfied by u = h00/r^2.
    
    Parameters:
        r: radial coordinate
        Y: state vector [u, u']
        p_r: pressure at this point
        rho_r: energy density at this point
        lamb_r: lambda at this point
        drho: derivative of energy density with respect to r
    
    Returns:
        [u', u''] derivative vector
    """
    u, up = Y
    pi = np.pi
    
    eL = np.exp(lamb_r)
    e2L = eL * eL
    e3L = e2L * eL
    
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3
    
    # A(r) coefficient
    A = (
        -1.0 - 3.0*eL + 3.0*e2L + e3L
        + 96.0*e2L*(-5.0 + 2.0*eL)*(pi**2)*r4*(p_r**2)
        + 512.0*e3L*(pi**3)*r6*(p_r**3)
        - 20.0*eL*(-1.0 + eL)*pi*r2*rho_r
        + 4.0*eL*pi*r2*p_r*(15.0 - 9.0*eL + 6.0*e2L - 40.0*eL*pi*r2*rho_r)
        + 8.0*eL*pi*r3*drho
    )
    
    # D(r) denominator
    D = r2 * (-1.0 + eL + 8.0*eL*pi*r2*p_r)
    if D == 0.0 or not np.isfinite(D):
        raise ZeroDivisionError(f"Denominator is singular at r={r}")
    
    # B(r) coefficient
    B = 1.0 + eL + 4.0*eL*pi*r2*p_r - 4.0*eL*pi*r2*rho_r
    
    # Expression for u''
    upp = (A/D)*u - ((4.0 + B)/r)*up - ((2.0 + 2.0*B)/r2)*u
    
    return np.array([up, upp], dtype=float)


def rk4_step(f, x, y, h, **kwargs):
    """
    Perform one RK4 step.
    
    Parameters:
        f: RHS function for the ODE, f(x, y, **kwargs)
        x: current independent variable
        y: current state vector
        h: step size
        **kwargs: extra parameters passed to f
    
    Returns:
        next state vector
    """
    k1 = f(x,           y,            **kwargs)
    k2 = f(x + 0.5*h,   y + 0.5*h*k1, **kwargs)
    k3 = f(x + 0.5*h,   y + 0.5*h*k2, **kwargs)
    k4 = f(x + h,       y + h*k3,     **kwargs)
    
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)



# ===== To Be Executed =======
def main():

    config_path = "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/"
    with open(f"{config_path}h_eps_config.pkl", "rb") as f:
        h_eps_dic = pickle.load(f)
    
    # print([type(h_eps_dic[i]) for i in h_eps_dic.keys()])
    h = float(h_eps_dic["h"])
    eps = float(h_eps_dic["eps"])
    d_num = h_eps_dic["d_num"]
    K_cgs = float(h_eps_dic["K_cgs"])
    print(f"\n===== config loaded: h = {h}, eps = {eps}, d_num = {d_num}, K_cgs = {K_cgs} =====\n")

    # Loading the TOV solution
    grid_path = f"/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/numpy_grid/h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}"
    
    with open(f"{grid_path}/Rgrid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl", "rb") as f:
        Rgrid = pickle.load(f)

    with open(f"{grid_path}/PMSolGrid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl", "rb") as f:
        PMSol = pickle.load(f)
        
    print("\n===== TOV Solution loaded =====\n")

    # Unit conversion before numerical integration
    PcArr_cgs = np.logspace(33.5, 36.5, d_num)

    # Convert to geometric units
    K, PcArr = cgs_to_geom(PcArr_cgs, 1, K_cgs)[:2]


    # Store h00 solutions for all models
    H00Grid = []

    print("\n===== Integration of H00 Starts =====\n")

    
    for nnn in tqdm(range(0, d_num), desc=f"H00 Solver", total=d_num):
        
        # Index of the current stellar model
        solToUseID = nnn
        PMSolArr = np.array(PMSol[solToUseID])
        
        # Extract background solution
        RArr = np.array(Rgrid[solToUseID])    # radial grid
        PArr = PMSolArr[:, 1]                  # pressure
        MArr = PMSolArr[:, 0]                  # mass
        
        # Defining the equation of states
        n1polytrope = PolytropeEOS_n1(K)
        
        # Compute energy density array
        RhoArr = n1polytrope.energy_density(PArr)
        
        # Compute Lambda array (metric function)
        # e^lambda = r / (r - 2M)
        LambArr = np.log(RArr / (RArr - 2*MArr))
        
        # Compute derivative of energy density w.r.t. r
        DrhoArr = np.gradient(RhoArr, RArr, edge_order=2)
        
        # Central pressure of the current model
        pc = PcArr[solToUseID]
        
        # Set initial conditions using analytic series expansion
        # a0 = 1 is the normalization constant; a8 = 0 ignores higher-order terms
        h00_ini = h00_approx(r=RArr[0], K=K, pc=pc, a0=1, a8=0)
        h00p_ini = h00p_approx(r=RArr[0], K=K, pc=pc, a0=1, a8=0)
        
        # Initialize solution array
        H00ini = np.array([h00_ini, h00p_ini])
        H00Sol = np.zeros((RArr[1:].shape[0], 2))
        H00Sol[0] = H00ini
        
        # Integrate outward using RK4
        for i, r in tqdm(enumerate(RArr[1:-2]), desc=f"Model {nnn+1}/{d_num}"):
            
            # Build background parameters at this point
            params = {
                "p_r": PArr[i+1],
                "rho_r": RhoArr[i+1],
                "lamb_r": LambArr[i+1],
                "drho": DrhoArr[i]
            }
            
            # Perform one RK4 step
            H00Sol[i+1] = rk4_step(h00pp_rhs, r, H00Sol[i], h, **params)
        
        # Save solution for this model
        H00Grid.append(H00Sol)
    print(f"Done! Computed {len(H00Grid)} models.")

    ## Saving H00 Grid to file
    with open(f"{grid_path}/H00grid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl", "wb") as f:
        pickle.dump(H00Grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n===== H00 Grid Saved =====\n")

    
    
    # ========= Solve the u equation (select a few models for validation) =======
    UGrid = []

    # print(f"Solve u = h00/r^2 for validation...")
    print("\n===== Integration of u := H00/r^2 Starts =====\n")
    for nnn in np.arange(0,d_num):
        
        solToUseID = nnn
        PMSolArr = np.array(PMSol[solToUseID])
        
        RArr = np.array(Rgrid[solToUseID])
        PArr = PMSolArr[:, 1]
        MArr = PMSolArr[:, 0]
        
        # Defining the equation of states
        n1polytrope = PolytropeEOS_n1(K)

        # Background arrays
        RhoArr = n1polytrope.energy_density(PArr)
        LambArr = np.log(RArr / (RArr - 2.0*MArr))
        DrhoArr = np.gradient(RhoArr, RArr, edge_order=2)
        
        pc = PcArr[solToUseID]
        
        # Initial conditions
        r0 = RArr[0]
        h0 = h00_approx(r=r0, K=K, pc=pc, a0=1, a8=0)
        hp0 = h00p_approx(r=r0, K=K, pc=pc, a0=1, a8=0)
        
        u0 = h0 / (r0**2)
        up0 = (r0*hp0 - 2.0*h0) / (r0**3)
        
        USol = np.zeros((RArr.shape[0], 2), dtype=float)
        USol[0] = np.array([u0, up0], dtype=float)
        
        # Integrate with adaptive step size
        for i, r in tqdm(enumerate(RArr[:-1]), total=len(RArr)-1, desc=f"Model {nnn+1}"):
            
            dr = RArr[i+1] - RArr[i]
            
            params = {
                "p_r": PArr[i],
                "rho_r": RhoArr[i],
                "lamb_r": LambArr[i],
                "drho": DrhoArr[i],
            }
            
            USol[i+1] = rk4_step(u_pp_rhs, r, USol[i], dr, **params)
        
        UGrid.append(USol)
    print(f"Done! Computed {len(H00Grid)} models.")

    ## Saving U00 Grid to file
    with open(f"{grid_path}/U00grid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl", "wb") as f:
        pickle.dump(UGrid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n===== U Grid Saved =====\n")

if __name__ == "__main__":
    main()