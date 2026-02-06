from pathlib import Path
import pickle

# ----- Most Important Coefficients -----
h = "1e-4"    # step size
eps = "1e-3"  # starting position
d_num = 25    # number of central pressures



# ----- Other Repetitively used Coefficients -----
K_cgs = "1.6e5" # polytrope coefficient in cgs unit
h_eps_dic = {"h": h, "eps": eps, "d_num": d_num, "K_cgs": K_cgs}

path = Path("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/h_eps_config.pkl")

with path.open("wb") as f:
    pickle.dump(h_eps_dic, f, protocol=pickle.HIGHEST_PROTOCOL)