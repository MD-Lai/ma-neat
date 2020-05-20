import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

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

def plot_series(seriess, labels=None, markers=None, title="", x_axis="",y_axis="", show=True, save_file=None):
    if labels is None:
        labels = [i for i in range(len(seriess))]
    if markers is None:
        markers = ["" for i in range(len(seriess))]
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    for (seriesx,seriesy),l,m in zip(seriess, labels, markers):
        ax.plot(seriesx, seriesy, label=l, marker=m)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

    ax.legend(loc='best')
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

def ind_peak_dist(series):
    ind, peak = max(list(enumerate(series)), key=lambda x: x[1])
    d_avg = ( sum( [(x_i-peak) ** 2 for x_i in series] ) / ( len(series)-1) )**0.5
    return ind, peak, d_avg

def ipds_tests(tst, bdt, runs, window):
    gbam = get_test_fitnesses(tst, bdt, runs)
    ipds = []
    for gens, best, avg, median in gbam:
        # print(best)
        best_smooth = smooth_series(best, window)
        ipd = ind_peak_dist(best_smooth)

        ipds.append(ipd)

    # print(ipds[:10])
    return ipds

def plot_ipds(ipds, labels=None, markers=None, title="", x_axis="", y_axis="", show=True, save_file=None): # flesh out to include labels
    if labels is None:
        labels = [i for i in range(len(ipds))]
    if markers is None:
        markers = ["." for i in range(len(ipds))]

    fig, ax = plt.subplots()
    ax.set_title(title)
    for ipd, l, m in zip(ipds, labels, markers):
        i, p, d = zip(*ipd)
        ax.scatter(i, p, label=l, marker=m, alpha=0.5)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
    ax.legend(loc="best")
    ax.grid()
    if show:
        plt.show()

    if save_file is not None:
        ax.savefig(save_file) 
    
    plt.close()