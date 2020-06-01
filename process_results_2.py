import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy import median
import pandas as pd
from math import ceil

# Need to track generations to completion, max fitness reached. 

def make_dir(tst, bdt):
    path = f"processed/test_{tst}_{bdt}"
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as osE:
        print ("Creation of the directory %s failed" % path)
        print (osE)

def open_test(tst, bdt, ind):
    pickle_file = f"spartan/test_{tst}_{bdt}/results_{tst}_{bdt}_{ind}.pickle"

    with open(pickle_file, 'rb') as data_pickle:
        winner, stats, b_stats, config = pickle.load(data_pickle)

    return winner, stats, b_stats, config

    

def plot_series(seriesxs, seriesys, labels=None, markers=None, title="", x_axis="",y_axis="", show=True, save_file=None):
    plt.close('all')
    if labels is None:
        labels = [i for i in range(len(seriesxs))]
    if markers is None:
        markers = ["" for i in range(len(seriesxs))]
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    for seriesx,seriesy,l,m in zip(seriesxs, seriesys, labels, markers):
        ax.plot(seriesx, seriesy, label=l, marker=m)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

    ax.legend(loc='best')
    ax.grid()
    ax.set_xlim(0,200)
    # ax.set_ylim(-30,25)
    if show:
        plt.show()

    if save_file is not None:
        fig.savefig(save_file)

    plt.close('all')

def get_fitness(tst,bdt,run):
    winner, stats, b_stats, config = open_test(tst, bdt, run)
    generation = range(len(stats.most_fit_genomes))
    best_fitness = np.array([c.fitness for c in stats.most_fit_genomes])
    avg_fitness = np.array(stats.get_fitness_mean())
    median_fitness = np.array(stats.get_fitness_median())

    return generation, best_fitness, avg_fitness, median_fitness

def get_test_fitnesses(tst, bdt, runs):
    run_l, run_h = runs
    fitnesses = []
    for i in range(run_l, run_h):
        f = get_fitness(tst, bdt, i)
        fitnesses.append(f)

    return fitnesses

def smooth_series(series, window):
    return [sum(series[i:i+window])/window for i in range(len(series)-window+1)]

# def ind_peak_dist_num(series, peak_scale=1, offset=0):
#     num_above = min(int(len(series) * (1-peak_scale)), len(series) - 1)
#     series_gen = sorted(enumerate(series), key=lambda x: x[1], reverse=True)

#     i, p = series_gen[num_above]

#     d_avg = ( sum( [(x_i-p) ** 2 for x_i in series] ) / ( len(series)-1) )**0.5
    
#     return i, p, d_avg

def ind_peak_line_dist_perc(series, peak_scale, offset):
    series_t = [s+offset for s in series]
    peak_scaled = max(series_t) * peak_scale
    for i, s_t in enumerate(series_t):
        if s_t >= peak_scaled:
            i = i 
            p = series[i]
            p_line = peak_scaled-offset
            d_avg = ( sum( [(x_i-p) ** 2 for x_i in series] ) / ( len(series)-1) )**0.5
            return i, p, p_line, d_avg
    # hasn't found it, all fitnesses < 0
    i, p = max(enumerate(series), key=lambda x: x[1])
    return i,p,p, ( sum( [(x_i-max(series)) ** 2 for x_i in series] ) / ( len(series)-1) )**0.5

# def ind_peak_dist_perc(series, peak_scale=1, offset=0):
#     # ind, peak = max(list(enumerate(series)), key=lambda x: x[1])
#     low = min(series) if max(series) <= 0 else 0 # only translate tests with fitnesses that peak < 0
#     series_t = [s-low for s in series]
#     peak_scaled = max(series_t) * peak_scale
#     for i, s_t in enumerate(series_t):
#         if s_t >= peak_scaled:
#             ind = i
#             break
    
#     peak = peak_scaled + low
#     d_avg = ( sum( [(x_i-peak) ** 2 for x_i in series] ) / ( len(series)-1) )**0.5
#     return ind, peak, d_avg

# def ind_peak_dist_min(series, peak_scale=1, offset=0):
#     # peak_scale does nothing but whatever
#     # Find fitness value that minimises distance value
#     avg = sum(series) / len(series)
#     dist_list = [ abs(x-avg) for x in series ] # distance from point x to avg 
#     i, p = min(enumerate(series), key=lambda x: dist_list[x[0]])
#     d_avg = sum(dist_list) / len(dist_list)
#     return i,p,d_avg

def demonstrate_cutoff(tst, bdt, run, window, peak_scale, offset):
    tst_names=["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    bdt_names=["Base","N_Prob","H_Prob","N_Eps","H_Eps","N_TS","H_TS"]
    _,f,_,_ = get_fitness(tst, bdt, run)
    f = smooth_series(f, window)
    i,p,l,d = ind_peak_line_dist_perc(f, peak_scale, offset)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title(f"Test:{tst_names[tst]} Bandit:{bdt_names[bdt]} Run:{run}")
    ax.plot(range(len(f)), f, label=f"Smoothed Fitness (window={window})")
    ax.plot(range(200), [l for _ in range(200)],label=f"{peak_scale:.2f}x Peak Fitness={l:.2f}")
    ax.scatter([i],[p], label=f"Selected Point : (Gens,Fit)=({i},{p:.2f})", marker='o', c="red")
    ax.set_xlim(0,200)
    ax.set_ylim(-30,25)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generation")
    ax.legend(loc="lower right")
    plt.show()
    fig.savefig(f"Test_{tst_names[tst]} Bandit_{bdt_names[bdt]} Run_{run}")

def iplds_runs(tst, bdt, runs, window, peak_scale=1, offset=0):
    gbam = get_test_fitnesses(tst, bdt, runs)
    iplds = []
    for gens, best, avg, median in gbam:
        # print(best)
        best_smooth = smooth_series(best, window)
        ipld = ind_peak_line_dist_perc(best_smooth, peak_scale, offset)
        iplds.append(ipld)

    # print(ipds[:10])
    return iplds

def iplds_bdts_runs(tst, bdts, runs, window, offset, peak_scale=1):
    return [iplds_runs(tst, b, runs, window, peak_scale, offset) for b in range(*bdts)]

def iplds_tsts_bdts_runs(tsts, bdts, runs, window, offsets, peak_scale=1):
    return [iplds_bdts_runs(t, bdts, runs, window, offsets[t], peak_scale) for t in range(*tsts)]

def plot_iplds(iplds, labels=None, markers=None, title="", x_axis="", y_axis="", figsize=(16,9),show=True, save_file=None): # flesh out to include labels
    plt.close('all')
    if labels is None:
        labels = [i for i in range(len(iplds))]
    if markers is None:
        markers = ["." for i in range(len(iplds))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ymin = float("inf")
    for ipld, lab, mar in zip(iplds, labels, markers):
        i, p, l, d = zip(*ipld)
        ax.scatter(i, p, label=lab, marker=mar, alpha=0.5)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        low = min(p)
        if ymin > low:
            ymin = low
    
    ax.set_xlim(0,200)
    # ax.set_ylim(min(0,ymin))
    ax.legend(loc="best")
    ax.grid()
    if show:
        plt.show()

    if save_file is not None:
        fig.savefig(save_file) 
    
    plt.close('all')

def plot_and_save_iplds():
    offs = (250,0,0,0,0,0,8000)
    ts = iplds_tsts_bdts_runs((0,7),(0,7),(0,32),20, offs, 0.97)
    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    for i, ti in enumerate(ts):
        plot_iplds(ti, labels=labs, title=f"Generations and Peak Fitness Test:{test_names[i]}", x_axis="Generations", y_axis="Peak Fitness", save_file=f"processed/gen_peak_{i}")


def calc_p_vals(base, *models):
    return [stats.kruskal(base, m)[1] for m in models]

def show_p_vals():
    offs = (250,0,0,0,0,0,8000)
    ts = iplds_tsts_bdts_runs((0,7),(0,7),(0,32),20, offs, 0.97)
    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    pv_gens_all= []
    pv_fits_all= []
    gens_median_all = []
    fits_median_all = []
    for t in ts:
        gens_curr = []
        fits_curr = []
        gens_median = []
        fits_median = []
        for b in t:
            g,f,_,_ = zip(*b)
            gens_curr.append(g)
            fits_curr.append(f)
            gens_median.append( median(g) )
            fits_median.append( median(f) )
            
        gens_median_all.append(gens_median)
        fits_median_all.append(fits_median)
        
        pv_gens = calc_p_vals(gens_curr[0], *gens_curr)
        pv_fits = calc_p_vals(fits_curr[0], *fits_curr)

        pv_gens_all.append(pv_gens)
        pv_fits_all.append(pv_fits)
    
    dts = pd.DataFrame()
    for i, (g, f, gm, fm) in enumerate(zip(pv_gens_all, pv_fits_all, gens_median_all, fits_median_all)):
        # print(f"{test_names[i]} Gens p")

        dt = pd.DataFrame()
        dt['test_name'] = [test_names[i]]

        for l,gl,fl,gml,fml in zip(labs, g,f,gm,fm):
            if not l=='Base':

                dt[f'median_g_{l}'] = [f'{ceil(gml)} ({gl:.2f})'] if gl > 0.01 else [f'{ceil(gml)} (<0.01)'] #[gml]
                # dt['diff_g'] = [g-gm[0] for g in gm]
                # dt[f'p_g_{l}'] = [gl]
                dt[f'median_f_{l}'] = [f'{fml:.2f} ({fl:.2f})'] if fl > 0.01 else [f'{fml:.2f} (<0.01)'] # [fml]
                # dt['diff_f'] = [f-fm[0] for f in fm]
                # dt[f'p_f_{l}'] = [fl]
            else:
                dt[f'median_g_{l}'] = [f'{ceil(gml)}'] #[gml]
                dt[f'median_f_{l}'] = [f'{fml:.2f}']
        #dts = pd.concat([dts, dt], axis=1)

        dts = dts.append(dt)
    print(dts)
    dts.to_csv("Results.csv", index=False)
        # print(dt)
        # print(f"Median={gm}")
        # print(f"Diff={[g-gm[0] for g in gm]}")
        # print(f"P={g}")
        # print()
        # print(f"{test_names[i]} Fits p")
        # print(f"Median={fm}")
        # print(f"Diff={[f-fm[0] for f in fm]}")
        # print(f"P={f}")

# For all tsts, 
# def generate_all_graphs( , show=True, save_files=None):
if __name__=="__main__":
    show_p_vals()