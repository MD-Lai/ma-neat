import pickle
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from numpy import median, mean
import pandas as pd
from math import ceil

import visualize

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

# TODO rewrite this s.t. offset is the "goal" fitness, see if it reaches it 
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

def demonstrate_cutoff(tst, bdt, run, window, peak_scale, offset, show=False):
    tst_names=["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    bdt_names=["Base","N-Prob","H-Prob","N-Eps","H-Eps","N-TS","H-TS"]
    _,f,_,_ = get_fitness(tst, bdt, run)
    f = smooth_series(f, window)
    i,p,l,d = ind_peak_line_dist_perc(f, peak_scale, offset)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title(f"Cutoff Test:{tst_names[tst]} Bandit:{bdt_names[bdt]} Run:{run}")
    ax.plot(range(len(f)), f, label=f"Smoothed Fitness (window={window})")
    ax.plot(range(200), [l for _ in range(200)],label=f"{peak_scale:.2f}x Peak Fitness={l:.2f}")
    ax.scatter([i],[p], label=f"Selected Point : (Gens,Fit)=({i},{p:.2f})", marker='o', c="red")
    # ax.set_xlim(0,200)
    # ax.set_ylim(-30,25)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generation")
    ax.legend(loc="lower right")
    if show:
        plt.show()
    fig.savefig(f"processed/test_{tst}_{bdt}/cutoff_{tst}_{bdt}_{run}")
    # fig.savefig(f"Test_{tst_names[tst]} Bandit_{bdt_names[bdt]} Run_{run}")

def plot_and_save_cutoffs(figsize):
    offs = (250,0,0,0,0,0,8000)
    tsts = range(0,7)
    bdts = range(0,7)
    runs = range(0,32)
    
    for off, tst in zip(offs, tsts):
        for bdt in bdts:
            for run in runs:
                demonstrate_cutoff(tst, bdt, run, 20, 0.97, off)

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

def plot_iplds(iplds, labels=None, markers=None, title="", x_axis="", y_axis="", figsize=None,show=True, save_file=None): # flesh out to include labels
    plt.close('all')
    if labels is None:
        labels = [i for i in range(len(iplds))]
    if markers is None:
        markers = ["." for i in range(len(iplds))]

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else: 
        fig, ax = plt.subplots()

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

def plot_and_save_iplds(figsize):
    offs = (250,0,0,0,0,0,8000)
    ts = iplds_tsts_bdts_runs((0,7),(0,7),(0,32),20, offs, 0.97)
    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    for i, ti in enumerate(ts):
        plot_iplds(ti, labels=labs, title=f"Generations and Selected Fitness Test:{test_names[i]}", x_axis="Generations", y_axis="Selected Fitness",figsize=figsize, save_file=f"processed/gen_peak_{i}")


def calc_p_vals(base, *models, alt=""):
    ps = []

    for m in models:
        # test base < m 
        # if 
        _,p = mannwhitneyu(base, m, alternative=alt) 
        ps.append(p)

    return ps

def show_p_vals(rel):
    offs = (250,0,0,0,0,0,8000)
    if rel:
        ts = iplds_tsts_bdts_runs((0,7),(0,7),(0,32),20, offs, 0.97)
        with open("ts.pickle", "wb") as t:
            pickle.dump(ts, t)
    else:
        with open("ts.pickle", "rb") as t:
            ts = pickle.load(t)

    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    pv_gens_all= []
    pv_fits_all= []
    gens_mean_all = []
    fits_mean_all = []
    for t in ts:
        gens_curr = []
        fits_curr = []
        gens_mean = []
        fits_mean = []
        for b in t:
            g,f,_,_ = zip(*b)
            gens_curr.append(g)
            fits_curr.append(f)
            gens_mean.append( mean(g) )
            fits_mean.append( mean(f) )
            
        gens_mean_all.append(gens_mean)
        fits_mean_all.append(fits_mean)
        
        pv_gens = calc_p_vals(gens_curr[0], *gens_curr, alt="greater") # base > models = H1 
        pv_fits = calc_p_vals(fits_curr[0], *fits_curr, alt="greater") # base < models = H1

        pv_gens_all.append(pv_gens)
        pv_fits_all.append(pv_fits)
    
    dts = pd.DataFrame()
    for i, (g, f, gm, fm) in enumerate(zip(pv_gens_all, pv_fits_all, gens_mean_all, fits_mean_all)):
        # print(f"{test_names[i]} Gens p")

        dt = pd.DataFrame()
        dt['test_name'] = [test_names[i]]

        for l,gl,fl,gml,fml in zip(labs, g,f,gm,fm):
            if not l=='Base':

                dt[f'mean_g_{l}'] = [f'{ceil(gml)} ({gl:.2f})'] if gl > 0.01 else [f'{ceil(gml)} (<0.01)'] #[gml]
                # dt['diff_g'] = [g-gm[0] for g in gm]
                # dt[f'p_g_{l}'] = [gl]
                dt[f'mean_f_{l}'] = [f'{fml:.2f} ({fl:.2f})'] if fl > 0.01 else [f'{fml:.2f} (<0.01)'] # [fml]
                # dt['diff_f'] = [f-fm[0] for f in fm]
                # dt[f'p_f_{l}'] = [fl]
            else:
                dt[f'mean_g_{l}'] = [f'{ceil(gml)}'] #[gml]
                dt[f'mean_f_{l}'] = [f'{fml:.2f}']
        #dts = pd.concat([dts, dt], axis=1)

        dts = dts.append(dt)
    print(dts)
    dts.to_csv("Results_mannwhitu.csv", index=False)

def show_mean_gens_fit_from_data(gen_fits, bdtnames, save=None,show=False):
    # print(list(zip(bdtnames, gen_fits)))
    gens = []
    fits = []
    for gfs in gen_fits:
        g,f = zip(*gfs)
        gens.append(g)
        fits.append(f)

    dts = pd.DataFrame()
    for b,g,f in zip(bdtnames,gens,fits):
        mean_g = mean(g)
        mean_f = mean(f)
        err_g = [max(g)+1,min(g)+1]
        err_f = [max(f),min(f)]

        n_thresh = len([x for x in g if x < 199])


        dt = pd.DataFrame()
        dt["MAB"] = [b]
        dt["Mean Generations"] = [round(mean_g)+1]
        dt["Mean Fitness"] = [f"{mean_f:.2f}"]
        dt["n Threshold"] = [n_thresh]
        dt["Range Generations"] = [err_g] 
        dt["Range Fitness"] = [err_f]

        dts = dts.append(dt)
    # print(mean_g, mean_f, err_g, err_f, p_g, p_f)
    dts.sort_values(by=['Mean Generations'], inplace=True)
    if save is not None:
        dts.to_csv(save, index=False)

    if show:
        print(dts)

def show_p_vals_krusk(rel):
    offs = (250,0,0,0,0,0,8000)
    if rel:
        ts = iplds_tsts_bdts_runs((0,7),(0,7),(0,32),20, offs, 0.97)
        with open("ts.pickle", "wb") as t:
            pickle.dump(ts, t)
    else:
        with open("ts.pickle", "rb") as t:
            ts = pickle.load(t)
    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    pv_gens_all= []
    pv_fits_all= []
    gens_mean_all = []
    fits_mean_all = []
    for t in ts:
        gens_curr = []
        fits_curr = []
        gens_mean = []
        fits_mean = []
        for b in t:
            g,f,_,_ = zip(*b)
            gens_curr.append(g)
            fits_curr.append(f)
            gens_mean.append( mean(g) )
            fits_mean.append( mean(f) )
            
        gens_mean_all.append(gens_mean)
        fits_mean_all.append(fits_mean)
        
        _, pv_gens = kruskal(*gens_curr) # base > models = H1 
        _, pv_fits = kruskal(*fits_curr) # base < models = H1

        pv_gens_all.append(pv_gens)
        pv_fits_all.append(pv_fits)
    
    dts = pd.DataFrame()
    for i, (g, f, gm, fm) in enumerate(zip(pv_gens_all, pv_fits_all, gens_mean_all, fits_mean_all)):
        # print(f"{test_names[i]} Gens p")

        dt = pd.DataFrame()
        dt['test_name'] = [test_names[i]]
        dt['p_value_g'] = [g]
        dt['p_value_f'] = [f]

        dts = dts.append(dt)
    print(dts)
    dts.to_csv("Results_kruskal.csv", index=False)

def show_pairwise_p_vals_mann_whitney_from_data(gen_fits, bdtnames, save=None, save2=None, show=False):
    """
    +---+---+---+---+
    |   |aaa|bbb|ccc|
    +---+---+---+---+
    |aaa|---|fab|fac|
    +---+---+---+---+
    |bbb|gba|---|fbc|
    +---+---+---+---+
    |ccc|gca|gcb|----
    +---+---+---+---+
    """
    dt = pd.DataFrame()
    dominance = {}
    for bdt in bdtnames:
        dominance[bdt] = {"gen_better": 0 , "gen_neutral":0, "gen_worse":0, "fit_better":0,"fit_neutral":0, "fit_worse":0, "sum_better":0}

    dt["names"] = bdtnames
    n_bdt = len(bdtnames)
    for x in range(n_bdt):
        p = []
        dom_curr = dominance[bdtnames[x]]
        g_x, f_x = zip(*gen_fits[x])
        for y in range(n_bdt):
            g_y, f_y = zip(*gen_fits[y])
            significance = 0 # 0 for neither, -1 for < 1 for ^ 
            if x == y:
                p.append("-")
            elif x > y: # bottom triangle, generations
                try:
                    _, p_g = mannwhitneyu(g_x, g_y, alternative="less")
                    if p_g > 0.05 and p_g < 0.95:
                        point = ""
                    else:
                        point, significance = ("<", -1) if p_g <= 0.05 else ("^", 1)
                    if p_g < 0.01:
                        s_p_g = "<0.01"
                    else:
                        s_p_g = f"{p_g:.2f}"
                    p.append(f"{s_p_g}{point}")
                except:
                    p.append("*")
                    significance = 0

                if significance == -1: # the current x is winning
                    dominance[bdtnames[x]]["gen_better"] += 1
                    dominance[bdtnames[y]]["gen_worse"] += 1
                    dominance[bdtnames[x]]["sum_better"] += 1
                elif significance == 0:
                    dominance[bdtnames[x]]["gen_neutral"] += 1
                    dominance[bdtnames[y]]["gen_neutral"] += 1
                elif significance == 1:
                    dominance[bdtnames[x]]["gen_worse"] += 1
                    dominance[bdtnames[y]]["gen_better"] += 1
                    dominance[bdtnames[y]]["sum_better"] += 1
                    
            else: # top triangle, fitness
                try:
                    _, p_f = mannwhitneyu(f_x, f_y, alternative="greater")
                    if p_f > 0.05 and p_f < 0.95:
                        point = ""
                    else:
                        point, significance = ("<", -1) if p_f <= 0.05 else ("^", 1)
                    if p_f < 0.01:
                        s_p_f = "<0.01"
                    else:
                        s_p_f = f"{p_f:.2f}"
                    p.append(f"{s_p_f}{point}")
                except:
                    p.append("*")
                    significance = 0

                if significance == -1: # the current x is winning
                    dominance[bdtnames[x]]["fit_better"] += 1
                    dominance[bdtnames[y]]["fit_worse"] += 1
                    dominance[bdtnames[x]]["sum_better"] += 1
                elif significance == 0:
                    dominance[bdtnames[x]]["fit_neutral"] += 1
                    dominance[bdtnames[y]]["fit_neutral"] += 1
                elif significance == 1:
                    dominance[bdtnames[x]]["fit_worse"] += 1
                    dominance[bdtnames[y]]["fit_better"] += 1
                    dominance[bdtnames[y]]["sum_better"] += 1

        dt[bdtnames[x]] = p
        # dominance[bdtnames[x]] = dom_temp

    dt = dt.T
    dom_dt = pd.DataFrame.from_dict(dominance).T
    dom_dt.sort_values(by=["sum_better"], ascending=False, inplace=True)
    # dom_dt = dom_dt.drop("sum_better")
    if save is not None:
        dt.to_csv(save)
    if save2 is not None:
        dom_dt.to_csv(save2)
    if show:
        print(dt)
        print(dom_dt)

def show_p_vals_krusk_from_data():
    pass
# TODO it would be great if we can see the bandits at every generation, generate rewards, plays, arms graphs. 
def plot_bdt(tst, bdt, run, show=False):
    winner, stats, b_stats, config = open_test(tst, bdt, run)
    
    #ov = overall
    arms, rewards, counts, gen_bests, gen_worsts, ov_bests, ov_worsts = list(zip(*b_stats.generational_data))
    arms = list(zip(*arms))
    rewards = list(zip(*rewards))
    counts = list(zip(*counts))

    labs = ["Base", "N-Prob", "H-Prob", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    arm_legend = ["node+", "node-", "node~", "conn+", "conn-", "conn~"]


    if bdt == 5 or bdt == 6:
        arms_temp = []
        for arm in arms:
            # print(arm)
            arm_t = [0 if s+f == 0 else s/(s+f) for s,f in arm]
            # print(arm_t)
            arms_temp.append(arm_t)
        arms = arms_temp
    
    fig = plt.figure(figsize=(10,6))

    if bdt == 0:
        ax_arm = fig.add_subplot(131)
        # heuristic numerical
        ax_rewh = fig.add_subplot(133)
        ax_rewn = fig.add_subplot(132)
        for i,rhn in enumerate(rewards):
            ax_rewh.plot([r[0] for r in rhn], label=arm_legend[i])
            ax_rewn.plot([r[1] for r in rhn], label=arm_legend[i])

        ax_rewn.set_title("Rewards (Numeric)")
        ax_rewn.set_xlabel("Generations")
        ax_rewn.set_ylabel("Cumulative Reward")
        ax_rewn.legend(loc="upper left")
        ax_rewn.grid()

        ax_rewh.set_title("Rewards (Heuristic)")
        ax_rewh.set_xlabel("Generations")
        ax_rewh.set_ylabel("Cumulative Reward")
        ax_rewh.legend(loc="upper left")
        ax_rewh.grid()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.0)

    else:
        ax_arm = fig.add_subplot(121)
        ax_rew = fig.add_subplot(122)
        for i,r in enumerate(rewards):
            ax_rew.plot(r, label=arm_legend[i])
        ax_rew.set_title("Rewards")
        ax_rew.set_xlabel("Generations")
        ax_rew.set_ylabel("Cumulative Reward")
        ax_rew.legend(loc="upper left")
        ax_rew.grid()

    for i,a in enumerate(arms):
        ax_arm.plot(a, label=arm_legend[i])
    add = " (mean)" if bdt == 5 or bdt == 6 else ""
    ax_arm.set_title(f"Arm Values{add}")
    ax_arm.set_xlabel("Generations")
    ax_arm.set_ylabel("Arm Value")
    ax_arm.legend(loc="upper left")
    ax_arm.grid()
    
    fig.suptitle(f"Bandit's Arms and Rewards over Generations Test:{test_names[tst]} Bandit:{labs[bdt]} Run:{run}")
    if show:
        plt.show()

    fig.savefig(f"processed/test_{tst}_{bdt}/bandit_{tst}_{bdt}_{run}")
    plt.close()
    plays_fig = plt.figure()
    plays = plays_fig.add_subplot(111)
    plays.set_title(f"Bandit Arms Play Count Test:{test_names[tst]} Bandit:{labs[bdt]} Run:{run}")
    for i,c in enumerate(counts):
        plays.plot(c, label=arm_legend[i])
    plays.legend()
    plays.set_xlabel("Generations")
    plays.set_ylabel("Play Count")
    if show:
        plt.show()
    plays_fig.savefig(f"processed/test_{tst}_{bdt}/banditcount_{tst}_{bdt}_{run}")
    plt.close()

def plot_all_bdt():
    tsts = range(0,7)
    bdts = range(0,7)
    runs = range(0,32)
    
    for tst in tsts:
        for bdt in bdts:
            for run in runs:
                plot_bdt(tst,bdt,run)

# TODO view the top 5 or so genomes throughout evolution
def view_top_n_networks(tst, bdt, run, n):
    winner, stats, b_stats, config = open_test(tst, bdt, run)
    top_n = [winner]
    top_n += stats.best_unique_genomes(n)
    
    for i, net in enumerate(top_n):
        file_prefix = f"processed/test_{tst}_{bdt}/net_{tst}_{bdt}_{run}_"
        filename = file_prefix + ("winner" if i == 0 else f"{i}")
        # print(filename)
        visualize.draw_net(config, net, view=False, filename=filename, show_disabled=True, fmt='png')
        os.remove(filename)

def view_all_test_networks():
    tsts = range(0,7)
    bdts = range(0,7)
    runs = range(0,32)
    
    for tst in tsts:
        for bdt in bdts:
            for run in runs:
                view_top_n_networks(tst,bdt,run,5)

if __name__=="__main__":
    _, rel = sys.argv
    show_p_vals(bool(int(rel)))

