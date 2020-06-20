import process_results_3 as p3 # for make_dir, open_test , get_fitness
import matplotlib.pyplot as plt
import pandas as pd
from numpy import median, mean, cumsum
from scipy.stats import kruskal, mannwhitneyu

bandit_names = ["Base", "N-Soft t=0.05", "H-Soft t=0.05", "N-Eps e=0.2", "H-Eps e=0.02", "N-TS", "H-TS"]
test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]

def show_p_vals_krusk():
    
    n_tests = 2
    n_bandits = 7
    
    gens_fits_all_tests = [p3.get_gens_fit(tst, range(n_bandits), range(32)) for tst in range(n_tests)]
    gens_means = []
    fits_means = []
    for gs,fs in gens_fits_all_tests:
        # print(gs)
        gens_means.append([mean(g) for g in gs])
        fits_means.append([mean(f) for f in fs])
    print(gens_means)
    print(fits_means)
    # fits_means = []
    # print(len(gens_fits_all_tests))
    # return
            
    
    dts = pd.DataFrame()
    # # for i, (g, f, gm, fm) in enumerate(zip(pv_gens_all, pv_fits_all, gens_mean_all, fits_mean_all)):
    #     # print(f"{test_names[i]} Gens p")

    #     dt = pd.DataFrame()
    #     dt['test_name'] = [test_names[i]]
    #     dt['p_value_g'] = [g]
    #     dt['p_value_f'] = [f]

    #     dts = dts.append(dt)
    # print(dts)
    # dts.to_csv("Results_kruskal.csv", index=False)