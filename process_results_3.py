import pickle
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from numpy import median, mean, cumsum
import pandas as pd
from math import ceil

import visualize
import process_results as p1

test_names=["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
bandit_names=["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]

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

def get_fitness_from_stats(stats):
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

# Not really necessary anymore, needs to be "select max fitness" or something
def demonstrate_cutoff(tst, bdt, run, window, peak_scale, offset, show=False):
    tst_names=test_names
    bdt_names=bandit_names
    _,f,_,_ = get_fitness(tst, bdt, run)
    f = smooth_series(f, window)
    i,p,l,d = ind_peak_line_dist_perc(f, peak_scale, offset)

    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title(f"Cutoff Test:{tst_names[tst]} Bandit:{bdt_names[bdt]} Run:{run}")
    ax.plot(range(len(f)), f, label=f"Smoothed Fitness (window={window})")
    ax.plot(range(len(f)), [l for _ in range(len(f))],label=f"{peak_scale:.2f}x Peak Fitness={l:.2f}")
    ax.scatter([i],[p], label=f"Selected Point : (Gens,Fit)=({i},{p:.2f})", marker='o', c="red")
    ax.set_xlim(0,len(f))
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
    labs = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    for i, ti in enumerate(ts):
        plot_iplds(ti, labels=labs, title=f"Generations and Selected Fitness Test:{test_names[i]}", x_axis="Generations", y_axis="Selected Fitness",figsize=figsize, save_file=f"processed/gen_peak_{i}", show=False)

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

    labs = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
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

def show_p_vals_krusk(tsts, bdts):
    # for all tests, assume 32 runs
    # for bandits from 0,bdth (because more is going to be added)
    # get list of terminal gens
    # get list of terminal fits
    # get kruskal p-val of gens
    # get kruskal p-val of fit 
    gens_tst_bdt = []
    fits_tst_bdt = []

    for t in range(tsts):
        gens_bdt = []
        fits_bdt = []

        for b in range(bdts):
            gens = []
            fits = []
            for r in range(32):
                try:
                    g,f = get_final_gen_fit(t,b,r)
                    gens.append(g)
                    fits.append(f)
                except:
                    pass
            gens_bdt.append(gens)
            fits_bdt.append(fits)

        gens_tst_bdt.append(gens_bdt)
        fits_tst_bdt.append(fits_bdt)

    # print(gens_tst_bdt)
    # print(fits_tst_bdt)
    bdt_names = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    # test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    
    for i, (gens, fits) in enumerate(zip(gens_tst_bdt, fits_tst_bdt)):
        print(kruskal(*gens))
        print(kruskal(*fits))

def get_gens_fit(tst, bdts, runs):
    gens_bdt_run = []
    fits_bdt_run = []
    for bdt in bdts:
        gens_run = []
        fits_run = []
        for run in runs:
            gens, fits, _,_ = get_fitness(tst, bdt, run)
            gens_run.append(gens[-1])
            fits_run.append(fits[-1])
        gens_bdt_run.append(gens_run)
        fits_bdt_run.append(fits_run)
    return gens_bdt_run, fits_bdt_run

def plot_fitness_scatter(tst, bdts, runs, show=True, save_file=None):
    labels = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    # test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    
    gens_bdt_run, fits_bdt_run = get_gens_fit(tst, bdts, runs)

    for bdt in bdts:
        gens_run = []
        fits_run = []
        for run in runs:
            gens, fits, _,_ = get_fitness(tst, bdt, run)
            gens_run.append(gens[-1])
            fits_run.append(fits[-1])
        gens_bdt_run.append(gens_run)
        fits_bdt_run.append(fits_run)
    
    fig, ax = plt.subplots()
    ax.set_title(f"Fitness Reached and Generations required Test:{test_names[tst]}")
    for g, f, l in zip(gens_bdt_run, fits_bdt_run, labels):
        ax.scatter(g,f,label=l, marker='.', alpha=0.5)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")

    ax.legend()
    ax.grid()
    if show:
        plt.show()
    if save_file is not None:
        fig.savefig(save_file)

    plt.close()

def plot_fitness_scatter_from_data(gen_fits, bdt_names, testname, save=None, show=False):
    fig, ax = plt.subplots(figsize=(9,6))
    # list of (generation, fitness) for each bandit 

    ax.set_title(f"Generations Required vs Fitness Achieved Test:{testname}")
    for gen_fit, bdt_name in zip(gen_fits, bdt_names):
        #[(g,f), (g,f),...], ["name","name",...]
        gen, fit = zip(*gen_fit)
        ax.scatter(gen, fit, label=bdt_name, alpha=0.5)
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")
    ax.set_xlim(-10,210)
    ax.grid()
    ax.legend(ncol=max(1,int(len(bdt_names)/7)))
    if show:
        plt.show()

    if save is not None:
        fig.savefig(save)
    plt.close()

def plot_gen_fit_boxplot_from_data(gen_fits, bdt_names, testname, save=None, show=False):
    gens = []
    fits = []

    for gfs in gen_fits:
        g,f = zip(*gfs)
        gens.append(g)
        fits.append(f)

    # min_g = min([min(gen) for gen in gens])
    # max_g = max([max(gen) for gen in gens])


    min_f = min([min(fit) for fit in fits])
    max_f = max([max(fit) for fit in fits])

    thresholded = [len([g for g in gen if g < 199]) for gen in gens]

    fig_bp, (ax_gens, ax_fits) = plt.subplots(1,2, figsize=(10,6), constrained_layout=True)
    ax_gens.boxplot(gens)
    ax_gens.set_xticklabels(bdt_names, rotation=90)
    ax_gens.set_ylabel("Generations Required")
    ax_gens.set_ylim(-5,210) # -2.5%,5% headroom
    for i,t in enumerate(thresholded): 
        ax_gens.text(i+1, 204, t, horizontalalignment='center') # label at 2.5% headroom

    ax_fits.boxplot(fits)
    ax_fits.set_xticklabels(bdt_names, rotation=90)
    ax_fits.set_ylabel("Fitness Achieved")
    ax_fits.set_ylim(min_f - (max_f-min_f)*0.025, max_f + (max_f-min_f)*0.05)
    for i,t in enumerate(thresholded): 
        ax_fits.text(i+1, max_f + (max_f-min_f) * 0.02, t, horizontalalignment='center') # label at 2.5% headroom

    fig_bp.suptitle(f"MAB Generations Required and Fitness Achieved Test:{testname}")

    if save is not None:
        fig_bp.savefig(save, bbox_inches="tight")

    if show:
        plt.show()


# TODO it would be great if we can see the bandits at every generation, generate rewards, plays, arms graphs. 
def plot_bdt(tst, bdt, run, show=False):
    make_dir(tst,bdt)
    winner, stats, b_stats, config = open_test(tst, bdt, run)
    
    #ov = overall
    arms, rewards, counts = list(zip(*b_stats.generational_data))
    arms = list(zip(*arms))
    rewards = list(zip(*rewards))
    counts = list(zip(*counts))

    labs = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    # test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    if len(arms) == 6:
        arm_legend = ["node+", "node-", "node~", "conn+", "conn-", "conn~"]
    elif len(arms) == 5: # len arms == 5
        arm_legend = ["node+", "node~", "conn+", "conn~", "cross"]
    else:
        arm_legend = [f"{i}" for i in range(len(arms))]


    if bdt == 0:
        # change to average, cumulative rewards 2x2, top row = numeric, bottom row = heuristic, increase graph size especially vertical height
        fig, ((ax_rew_avgn, ax_rew_cumn), (ax_rew_avgh, ax_rew_cumh)) = plt.subplots(2,2, figsize=(12, 9), constrained_layout=True)
        # fig = plt.figure(figsize=(10,6))

        # ax_arm = fig.add_subplot(131)
        # heuristic numerical
        # ax_rewn = fig.add_subplot(132)
        # ax_rewh = fig.add_subplot(133)
        for i, (rnh,count) in enumerate(zip(rewards, counts)):
            ax_rew_avgn.plot([rnp/(rnp+rnn or 1) for (rnp, rnn), (_,_) in rnh], label=arm_legend[i])
            ax_rew_cumn.plot([rnp/(rnp+rnn or 1) * c for ((rnp, rnn), (_,_)), c in zip(rnh, count)], label=arm_legend[i])
            
            ax_rew_avgh.plot([rhp/(rhp+rhn or 1) for (_,_), (rhp, rhn) in rnh], label=arm_legend[i])
            ax_rew_cumh.plot([rhp/(rhp+rhn or 1) * c for ((_, _), (rhp, rhn)), c in zip(rnh, count)], label=arm_legend[i])

        ax_rew_avgn.set_title("Average Rewards (Numeric)")
        ax_rew_avgn.set_xlabel("Generations")
        ax_rew_avgn.set_ylabel("Reward")
        ax_rew_avgn.legend(loc="upper left")
        ax_rew_avgn.grid()

        ax_rew_avgh.set_title("Average Rewards (Heuristic)")
        ax_rew_avgh.set_xlabel("Generations")
        ax_rew_avgh.set_ylabel("Reward")
        ax_rew_avgh.legend(loc="upper left")
        ax_rew_avgh.grid()

        ax_rew_cumn.set_title("Cumulative Rewards (Numeric)")
        ax_rew_cumn.set_xlabel("Generations")
        ax_rew_cumn.set_ylabel("Reward")
        ax_rew_cumn.legend(loc="upper left")
        ax_rew_cumn.grid()

        ax_rew_cumh.set_title("Cumulative Rewards (Heuristic)")
        ax_rew_cumh.set_xlabel("Generations")
        ax_rew_cumh.set_ylabel("Reward")
        ax_rew_cumh.legend(loc="upper left")
        ax_rew_cumh.grid()

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=0)

    else:
        fig, (ax_rew_avg, ax_rew_cum) = plt.subplots(1,2, figsize=(10,6), constrained_layout=True)
        # ax_arm = fig.add_subplot(121)
        # ax_rew = fig.add_subplot(122)
        for i, (r,count) in enumerate(zip(rewards, counts)):
            ax_rew_avg.plot([rp/(rp+rn or 1) for (rp,rn) in r], label=arm_legend[i])
            ax_rew_cum.plot([rp/(rp+rn or 1) * c for (rp,rn),c in zip(r,count)], label=arm_legend[i])

        ax_rew_avg.set_title(f"Average Reward")
        ax_rew_avg.set_xlabel("Generations")
        ax_rew_avg.set_ylabel("Reward")
        ax_rew_avg.legend(loc="upper left")
        ax_rew_avg.grid()

        ax_rew_cum.set_title("Cumulative Reward")
        ax_rew_cum.set_xlabel("Generations")
        ax_rew_cum.set_ylabel("Reward")
        ax_rew_cum.legend(loc="upper left")
        ax_rew_cum.grid()
    
    fig.suptitle(f"Bandit's Average and Cumulative Reward over Generations Test:{test_names[tst]} Bandit:{labs[bdt]} Run:{run}")
    if show:
        plt.show()

    fig.savefig(f"processed/test_{tst}_{bdt}/bandit_{tst}_{bdt}_{run}")
    plt.close()
    plays_fig, plays = plt.subplots()
    # plays = plays_fig.add_subplot(111)
    plays.set_title(f"Bandit Arms Play Count Test:{test_names[tst]} Bandit:{labs[bdt]} Run:{run}")
    for i,c in enumerate(counts):
        plays.plot(c, label=arm_legend[i])
    plays.legend()
    plays.set_xlabel("Generations")
    plays.set_ylabel("Play Count")
    plays.grid()
    if show:
        plt.show()
    plays_fig.savefig(f"processed/test_{tst}_{bdt}/banditcount_{tst}_{bdt}_{run}")
    plt.close()

def plot_bdt_from_b_stats(b_stats, testname, banditname, runname, savereward=None, savecount=None, showreward=False, showcount=False):

    arms, rewards, counts = list(zip(*b_stats.generational_data))
    arms = list(zip(*arms))
    rewards = list(zip(*rewards))
    counts = list(zip(*counts))

    labs = bandit_names # ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
    # test_names = test_names # ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    if len(arms) == 6:
        arm_legend = ["node+", "node-", "node~", "conn+", "conn-", "conn~"]
    elif len(arms) == 5: # len arms == 5
        arm_legend = ["node+", "node~", "conn+", "conn~", "cross"]
    else:
        arm_legend = [f"{i}" for i in range(len(arms))]

    if "Static" in banditname: # static bandit, plot heuristic and numerical
        # change to average, cumulative rewards 2x2, top row = numeric, bottom row = heuristic, increase graph size especially vertical height
        fig, ((ax_rew_avgn, ax_rew_cumn), (ax_rew_avgh, ax_rew_cumh)) = plt.subplots(2,2, figsize=(12, 9), constrained_layout=True)
        # fig = plt.figure(figsize=(10,6))

        # ax_arm = fig.add_subplot(131)
        # heuristic numerical
        # ax_rewn = fig.add_subplot(132)
        # ax_rewh = fig.add_subplot(133)
        for i, (rnh,count) in enumerate(zip(rewards, counts)):
            ax_rew_avgn.plot([rnp/(rnp+rnn or 1) for (rnp, rnn), (_,_) in rnh], label=arm_legend[i])
            ax_rew_cumn.plot([rnp/(rnp+rnn or 1) * c for ((rnp, rnn), (_,_)), c in zip(rnh, count)], label=arm_legend[i])
            
            ax_rew_avgh.plot([rhp/(rhp+rhn or 1) for (_,_), (rhp, rhn) in rnh], label=arm_legend[i])
            ax_rew_cumh.plot([rhp/(rhp+rhn or 1) * c for ((_, _), (rhp, rhn)), c in zip(rnh, count)], label=arm_legend[i])

        ax_rew_avgn.set_title("Average Rewards (Numeric)")
        ax_rew_avgn.set_xlabel("Generations")
        ax_rew_avgn.set_ylabel("Reward")
        ax_rew_avgn.legend(loc="upper left")
        ax_rew_avgn.grid()

        ax_rew_avgh.set_title("Average Rewards (Heuristic)")
        ax_rew_avgh.set_xlabel("Generations")
        ax_rew_avgh.set_ylabel("Reward")
        ax_rew_avgh.legend(loc="upper left")
        ax_rew_avgh.grid()

        ax_rew_cumn.set_title("Cumulative Rewards (Numeric)")
        ax_rew_cumn.set_xlabel("Generations")
        ax_rew_cumn.set_ylabel("Reward")
        ax_rew_cumn.legend(loc="upper left")
        ax_rew_cumn.grid()

        ax_rew_cumh.set_title("Cumulative Rewards (Heuristic)")
        ax_rew_cumh.set_xlabel("Generations")
        ax_rew_cumh.set_ylabel("Reward")
        ax_rew_cumh.legend(loc="upper left")
        ax_rew_cumh.grid()

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=0)

    else:
        fig, (ax_rew_avg, ax_rew_cum) = plt.subplots(1,2, figsize=(10,6), constrained_layout=True)
        # ax_arm = fig.add_subplot(121)
        # ax_rew = fig.add_subplot(122)
        for i, (r,count) in enumerate(zip(rewards, counts)):
            ax_rew_avg.plot([rp/(rp+rn or 1) for (rp,rn) in r], label=arm_legend[i])
            ax_rew_cum.plot([rp/(rp+rn or 1) * c for (rp,rn),c in zip(r,count)], label=arm_legend[i])

        ax_rew_avg.set_title(f"Average Reward")
        ax_rew_avg.set_xlabel("Generations")
        ax_rew_avg.set_ylabel("Reward")
        ax_rew_avg.legend(loc="upper left")
        ax_rew_avg.grid()

        ax_rew_cum.set_title("Cumulative Reward")
        ax_rew_cum.set_xlabel("Generations")
        ax_rew_cum.set_ylabel("Reward")
        ax_rew_cum.legend(loc="upper left")
        ax_rew_cum.grid()

    fig.suptitle(f"Bandit's Average and Cumulative Reward over Generations Test:{testname} Bandit:{banditname} Run:{runname}")
    
    if savereward is not None:
        fig.savefig(savereward)

    if showreward:
        plt.show()

    plt.close()

    plays_fig, ax_plays = plt.subplots(figsize=(8,6))
    plays_fig.suptitle(f"Bandit Arms Play Count Test:{testname} Bandit:{banditname} Run:{runname}")
    # plays = plays_fig.add_subplot(111)
    # ax_arms.set_title("Normalised reward")
    # for i,a in enumerate(arms):
    #     ax_arms.plot(a, label=arm_legend[i])
    # ax_arms.legend()
    # ax_arms.set_xlabel("Generations")
    # ax_arms.set_ylabel("Arm Value")
    # ax_arms.grid()

    ax_plays.set_title("Play count")
    for i,c in enumerate(counts):
        ax_plays.plot(c, label=arm_legend[i])
    ax_plays.legend()
    ax_plays.set_xlabel("Generations")
    ax_plays.set_ylabel("Play Count")
    ax_plays.grid()

    if savecount is not None:
        plays_fig.savefig(savecount)
    
    if showcount:
        plt.show()
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
        visualize.draw_net(config, net, view=False, filename=filename, show_disabled=False, fmt='png')
        os.remove(filename)

def plot_top_n_networks_from_data(winner, stats, cfg, n, save=None, show=False):
    top_n = [winner]
    top_n += stats.best_unique_genomes(n)
    
    file_prefix = f"{save}_" if save is not None else "net_"
    for i, net in enumerate(top_n):
        filename = file_prefix + ("winner" if i == 0 else f"{i}")
        # print(filename)
        visualize.draw_net(cfg, net, view=show, filename=filename, show_disabled=False, fmt='png')
        os.remove(filename)
        
        

def view_all_test_networks():
    tsts = range(0,7)
    bdts = range(0,7)
    runs = range(0,32)
    
    for tst in tsts:
        for bdt in bdts:
            for run in runs:
                view_top_n_networks(tst,bdt,run,5)

# Function that gets fitness at last generation
def get_final_gen_fit(tst, bdt, run):
    generation, best_fitness, avg_fitness, median_fitness = get_fitness(tst, bdt, run)
    g = generation[-1]
    win = best_fitness[-1]

    return g, win

def process_test(tst, bdt, run, *other_params):
    show = other_params[0]
    plot_bdt(tst,bdt,run,show=show)
    view_top_n_networks(tst,bdt,run,5)


if __name__=="__main__":
    _, rel = sys.argv
    show_p_vals(bool(int(rel)))

