import sys
import pickle
import visualize
import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np

_, tst, bdt, ind = sys.argv

rando = ["0"]
probs = ["1","2"]
epsil = ["3","4"]
betas = ["5","6"]

pickle_file = f"spartan/test_{tst}_{bdt}/results_{tst}_{bdt}_{ind}.pickle"


iarm = 0
ireward = 1
icount = 2
igen_best = 3
igen_worst = 4
iov_best = 5
iov_worst = 6

# TODO DONE Generate n-generations to fitness convergence 
# TODO DONE Generate peak avg fitness with moving window of 5 (for stagnation purposes)
# TODO DONE wrap it up in a function
# (winner, stats, b_stats, config)
# b_stats.generational_data = [[bdt.arms, bdt.rewards, bdt.counts, bdt.gen_best, bdt.gen_worst, bdt.best, bdt.worst], ...]
with open(pickle_file, 'rb') as data_pickle:
    winner, stats, b_stats, config = pickle.load(data_pickle)


arms, rewards, counts, gen_bests, gen_worsts, ov_bests, ov_worsts = list(zip(*b_stats.generational_data))

best_fitness = [c.fitness for c in stats.most_fit_genomes]
avg_fitness = stats.get_fitness_mean()

window_size = 20 # matches stagnation value
best_smooth = [sum(best_fitness[i:i+window_size])/window_size for i in range(len(best_fitness)-window_size)] # moving average
avg_smooth  = [sum(avg_fitness[i:i+window_size]) /window_size for i in range(len(avg_fitness)-window_size)]
rmsd = [(sum([(x_i-x__)**2 for x_i in best_fitness[i:i+window_size]])/(window_size-1))**0.5 for i,x__ in enumerate(best_smooth)]
fig, (fit,dev) = plt.subplots(1,2)
fit.set_title(f"fitness {tst} {bdt} {ind}")
fit.plot(best_fitness)
fit.plot(best_smooth)
dev.set_title(f"divergence {tst} {bdt} {ind}")
dev.plot(rmsd)
plt.show()
plt.close()
exit()

# Imagine a line, how high is it before it cuts off 90% of best genomes
cut = 0.9
line = sorted(best_smooth, reverse=True)[int(len(best_smooth)*(1-cut))]
above_line = [(1 if f > line else 0) for f in best_smooth]
for i,f_s in enumerate(above_line):
    if f_s > 0:
        print(i)
        break
print(best_smooth)
print(sum(above_line))
print(above_line)
print(i)

plt.figure(figsize=(16,9))
plt.title(f"Test={tst}, Bandit={bdt} Run={ind}")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.plot(best_smooth, label="fitness")
plt.plot([line for _ in best_smooth], label="cutoff", marker="_")
plt.plot(i, best_smooth[i],'ro')
plt.legend()
plt.show()

exit()

per_arms = list(zip(*arms))
per_rewards = list(zip(*rewards))
per_counts = list(zip(*counts))

fig1, arms = plt.subplots()
# betas needs different way to show arms
# Show expected reward: 1/(1+(f/s))
if bdt == "5" or bdt == "6": 
    for i,a in enumerate(per_arms):
        arms.plot([1/(1+((f+1)/(s+1))) for f,s in a], label = i)
else:
    for i, a in enumerate(per_arms):
        arms.plot(a, label=i)
arms.legend()

plt.show()

# rando needs different way to show rewards
# show both heuristic and exact rewards
if bdt == "0":
    fig2, (heur, exac) = plt.subplots(1,2, sharey=True)
    for i,h_e in enumerate(per_rewards):
        heur.plot([h for h,_ in h_e], label=i)
        exac.plot([e for _,e in h_e], label=i)
    heur.title.set_text("Heuristic")
    exac.title.set_text("Exact")
    heur.legend()
    exac.legend()
else:
    fig2, rewa = plt.subplots()
    for i,r in enumerate(per_rewards):
        rewa.plot(r, label=i)
    rewa.legend()
plt.show()

# plt.show()
# arms = [a[iarm] for a in b_stats.generational_data]
# rewards = [a[ireward] for a in b_stats.generational_data]
# counts = [a[icount] for a in b_stats.generational_data]

# if tst == "0": # random, 2 rewards (heuristic, actual)
path = f"processed/test_{tst}_{bdt}"
try:
    if not os.path.exists(path):
        os.mkdir(path)
except OSError as osE:
    print ("Creation of the directory %s failed" % path)
    print (osE)

visualize.plot_stats(stats, filename=f"{path}/processed_{tst}_{bdt}_{ind}_stats")
visualize.plot_species(stats, filename=f"{path}/processed_{tst}_{bdt}_{ind}_species")
visualize.draw_net(config, winner, filename=f"{path}/processed_{tst}_{bdt}_{ind}_winner")
visualize.draw_net(config, stats.best_genome(), filename=f"{path}/processed_{tst}_{bdt}_{ind}_winner")

# visualize.draw_net(config, winner, view=False, filename=test_prefix+"winner")
#visualize.draw_net(config, stats.best_genome(), view=False, filename=test_prefix+"best")
#visualize.plot_stats(stats, filename=test_prefix+"fitness")
#visualize.plot_species(stats, filename=test_prefix+"speciation")