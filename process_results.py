import sys
import pickle
import visualize
import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np

bandit_names = ["Base", "N-Soft", "H-Soft", "N-Eps", "H-Eps", "N-TS", "H-TS"]
test_names = ["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]


def get_gens(tst,bdt,ind, cut=0.1, window=5):
    pickle_file = f"spartan/test_{tst}_{bdt}/results_{tst}_{bdt}_{ind}.pickle"

    with open(pickle_file, 'rb') as data_pickle:
        winner, stats, b_stats, config = pickle.load(data_pickle)

    best_fitness = [c.fitness for c in stats.most_fit_genomes]

    window_size = window
    best_smooth = [sum(best_fitness[i:i+window_size])/window_size for i in range(len(best_fitness)-window_size)] # moving average

    # Calculate the fitness where only {cut} proportion of generations are above
    line = sorted(best_smooth, reverse=True)[int(len(best_smooth)*(cut))]
    above_line = [(1 if f >= line else 0) for f in best_smooth]

    for i,f_s in enumerate(above_line):
        if f_s > 0:
            return i

def get_all_gens(tst,bdt,low,rng, cut=0.1, window=5):
    return [get_gens(tst, bdt, i, cut, window) for i in range(low,low+rng)]

#p.plot_series([sorted(p.get_all_gens(3,bdt,0,32),reverse=True) for bdt in range(7)], labels=["rand","soft","Hsoft","eps","Heps","ts","Hts"], title="Cutoff fitnesses", y_label="generations", x_label="run (sorted by generations)")
def plot_series(series, labels=None, title="", x_label="",y_label=""):
    if labels is None:
        labels = [i for i in range(len(series))]
    
    fig, ax = plt.subplots()
    for s,l in zip(series, labels):
        ax.plot(s, label=l)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()

def make_dir(tst, bdt):
    path = f"processed/test_{tst}_{bdt}"
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as osE:
        print ("Creation of the directory %s failed" % path)
        print (osE)

# add point for max fitness or something
def plot_fitness(tst,bdt,ind, figsize=None, save=True, show=False):
    
    pickle_file = f"spartan/test_{tst}_{bdt}/results_{tst}_{bdt}_{ind}.pickle"

    with open(pickle_file, 'rb') as data_pickle:
        winner, statistics, b_stats, config = pickle.load(data_pickle)
    
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    median_fitness = np.array(statistics.get_fitness_median())
    
    if figsize is not None:
        fig, fits = plt.subplots(figsize=(16,9))
    else:
        fig, fits = plt.subplots()

    fits.plot(generation, best_fitness, 'r-', label="best")
    # fits.plot(generation, median_fitness, 'g-', label="median")
    fits.plot(generation, avg_fitness, 'b-', label="average")
    
    tst_names=test_names#["Pendulum_v0","BipedalWalker_v3","BipedalWalkerHardcore_v3","LunarLanderContinuous_v2","Banknote_Auth","Wine_Quality","MNIST"]
    bdt_names=bandit_names#["Base","N-Prob","H-Prob","N-Eps","H-Eps","N-TS","H-TS"]

    fits.set_title(f"Fitness Test:{tst_names[tst]} Bandit:{bdt_names[bdt]} Run:{ind}")
    fits.set_xlabel("Generations")
    fits.set_ylabel("Fitness")
    fits.grid()
    fits.legend(loc="lower right")

    if save:
        path = f"processed/test_{tst}_{bdt}"
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except OSError as osE:
            print (f"Creation of the directory {path} failed")
            print (osE)

        visualize.plot_species(statistics, graph_title=f"Speciation Test:{tst_names[tst]} Bandit:{bdt_names[bdt]} Run:{ind}", view=False, filename=f"processed/test_{tst}_{bdt}/species_{tst}_{bdt}_{ind}")

        filename = f"processed/test_{tst}_{bdt}/fitness_{tst}_{bdt}_{ind}"
        fig.savefig(filename)
    
    if show:
        plt.show()
    plt.close('all')

def plot_fitness_from_list(generation, best_fitness, avg_fitness, testname, banditname, runname, save=None, show=False):
    fig, fits = plt.subplots()
    fits.plot(generation, best_fitness, 'r-', label="best")
    fits.plot(generation, avg_fitness, 'b-', label="average")

    fits.set_title(f"Fitness Test:{testname} Bandit:{banditname} Run:{runname}")
    fits.set_xlabel("Generations")
    fits.set_ylabel("Fitness")
    fits.grid()
    fits.legend(loc="lower right")

    if save is not None:
        fig.savefig(save)
    
    if show:
        plt.show()
        
    plt.close('all')

def plot_species_from_stats(stats, testname, banditname, runname, save=None, show=False):
    visualize.plot_species(stats, graph_title=f"Speciation Test:{testname} Bandit:{banditname} Run:{runname}", view=show, filename=save)

def plot_and_save_fitnesses(tsts, bdts, figsize):
    # Assume all tests are just 0-31 for now
    for t in tsts:
        for b in bdts:
            for i in range(32):
                make_dir(t,b)
                plot_fitness(t,b,i,figsize=figsize, save=True,show=False)

# Just imagine all bandits are exactly the same for now, 
# add exceptions later
# def plot_bandits(tst,bdt,ind, save=True, show=False):

#     pickle_file = f"spartan/test_{tst}_{bdt}/results_{tst}_{bdt}_{ind}.pickle"

#     with open(pickle_file, 'rb') as data_pickle:
#         winner, stats, b_stats, config = pickle.load(data_pickle)s

#     arms, rewards, counts, gen_bests, gen_worsts, ov_bests, ov_worsts = list(zip(*b_stats.generational_data))

#     per_arms = list(zip(*arms))
#     per_rewards = list(zip(*rewards))

#     fig1, arms,rewa = plt.subplots(1,2, figsize=(16,9))
#     # betas needs different way to show arms
#     # Show expected reward: 1/(1+(f/s))
#     if bdt == "5" or bdt == "6": 
#         for i,a in enumerate(per_arms):
#             arms.plot([1/(1+((f+1)/(s+1))) for f,s in a], label = i)
#     else:
#         for i, a in enumerate(per_arms):
#             arms.plot(a, label=i)
#     arms.legend()

#     # rando needs different way to show rewards
#     # show both heuristic and exact rewards
#     # TODO Consider if we want to save heuristic and exact on the same or just 1 for simplicity
#     # TODO like [arms, heur/exac] as the subplot layout
#     if bdt == "0":
#         # fig2, (heur, exac) = plt.subplots(1,2, sharey=True)
#         for i,h_e in enumerate(per_rewards):
#             heur.plot([h for h,_ in h_e], label=i)
#             exac.plot([e for _,e in h_e], label=i)
#         heur.title.set_text("Heuristic")
#         exac.title.set_text("Exact")
#         heur.legend()
#         exac.legend()
#     else:
#         for i,r in enumerate(per_rewards):
#             rewa.plot(r, label=i)
#         rewa.legend()

#     if save:
#         pass

#     if show:
#         plt.show()