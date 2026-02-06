from tov_series_aux import p_approx, rho_approx, m_approx, nu_approx, lambda_approx
from EOS_aux import PolytropeEOS_n1
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os


K_cgs = 1.6e5
polytrope = PolytropeEOS_n1(K=K_cgs)

# The set of TOV equation and the conservation equation
def tov_rhs(r, y, eos, K):
    m, p = y[0], y[1]
    
    rhoEnergy = eos(p)

    dm_dr = 4 * np.pi * r**2 * rhoEnergy
    dp_dr = - (rhoEnergy + p) * (m + 4*np.pi*(r**3)*p) / (r * (r - 2*m))
    
    return [dm_dr, dp_dr]

def rk4_step(func, r, y, h, eos=polytrope.energy_density, K=K_cgs):

    k1 = np.array(func(r, y, eos, K))
    k2 = np.array(func(r + h/2, y + h*k1/2, eos, K))
    k3 = np.array(func(r + h/2, y + h*k2/2, eos, K))
    k4 = np.array(func(r + h, y + h*k3, eos, K))

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def nu_rhs(r, y, p, m):
    rr = 2 * (m + 4*np.pi*(r**3) * p) / (r*(r-2*m))
    return rr

# the new rk4 function for computing \nu numerically (with 6 params, so the previous is not overridden)
def rk4_step_nu(func, r, y, h, p, m):
    k1 = np.array(func(r, y, p, m))
    k2 = np.array(func(r + h/2, y + h*k1/2, p, m))
    k3 = np.array(func(r + h/2, y + h*k2/2, p, m))
    k4 = np.array(func(r + h, y + h*k3, p, m))
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# Unit Conversion

G_cgs = 6.67430e-8     # cm^3 g^-1 s^2
c_cgs = 2.99792458e10  # cm s^-1
M_sun_cgs = 1.989e33   # g
cm2km = 1e5

def cgs_to_geom(p_cgs, n, K_cgs, eos=polytrope.energy_density):

    # km^-2
    p_geom = p_cgs * G_cgs / c_cgs**4 * cm2km**2            
    # km^(2/n)
    K_geom = K_cgs / G_cgs**(1/n) / c_cgs**(2-2/n) / cm2km**(2/n)               
    
    # Polytropic EOS, km^-2
    # Use K_geom for the EnergyEOS mapping (same as notebook logic)
    rho_geom = ((1.0 + np.sqrt(4.0 * K_geom * p_geom))**2 - 1.0) / (4.0 * K_geom)

    return K_geom, p_geom, rho_geom

# The input mass has unit km in geometric units, and we want to express it in solar mass
def mass_g2Msun(m_km):
    m_cgs = m_km * cm2km
    m_msun = m_cgs / G_cgs * (c_cgs**2) / M_sun_cgs

    return m_msun



def main():

    with open("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/h_eps_config.pkl", "rb") as f:
        h_eps_dic = pickle.load(f)

    
    eps = float(h_eps_dic["eps"])
    h = float(h_eps_dic["h"])
    d_num = h_eps_dic["d_num"]

    SolArr = []
    RArr = []

    for i, P in enumerate(tqdm(np.logspace(33.5, 36.5, d_num), desc="TOV models", unit="model")):
        
        K, p_c, rho_c = cgs_to_geom(P, n=1, K_cgs=K_cgs, eos=polytrope.energy_density)
        
        print(f"model {i + 1}: ", "K = ", K, "p_c = ", p_c, "rho_c = ", rho_c) # Can confirm that this matches
        # break

        eos_local = PolytropeEOS_n1(K=K)
        if i == 0:
            print("DEBUG first model:")
            print("  P_cgs =", P)
            print("  K_geom =", K)
            print("  p_c =", p_c)
            print("  rho_c =", rho_c)
        
        # Initial Condition
        p0 = p_approx(eps, K, p_c)
        m0 = m_approx(eps, K, p_c)

        Sol = [np.array([m0,p0])]
        R_l = [eps]

        while (np.log10(Sol[0][1]/Sol[-1][1]) <= 10.1):
            newStep = rk4_step(tov_rhs, R_l[-1], Sol[-1], h, eos_local.energy_density, K)
            Sol.append(newStep)
            R_l.append(R_l[-1] + h)

        SolArr.append(Sol)
        RArr.append(R_l)
        if i == 0:
            print("DEBUG first model surface:")
            print("  R_surface =", R_l[-2])
            print("  M_surface_km =", Sol[-2][0])
            print("  M_surface_Msun =", mass_g2Msun(Sol[-2][0]))
    
    target_path = "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/numpy_grid/" + "h" + h_eps_dic["h"] + "_eps" + h_eps_dic["eps"]

    os.makedirs(target_path, exist_ok=True)

    target_file_PMsol = target_path + f"/PMSolGrid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl"

    target_file_RArr = target_path + f"/RGrid_h{h_eps_dic["h"]}_eps{h_eps_dic["eps"]}.pkl"

    print("Writing PMSol...")
    with open(target_file_PMsol, "wb") as f:                        
        pickle.dump(SolArr, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!!")
    
    print("Writing RGrid...")
    with open(target_file_RArr, "wb") as f:                        
        pickle.dump(RArr, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!!")
    
if __name__ == "__main__":
    main()
    
