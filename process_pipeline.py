import process_results as p1
import process_results_2 as p2
import process_results_3 as p3

import string
import random

# p1.plot_and_save_fitnesses(range(0,7), range(0,7), None) # Fitnesses
# # p2.plot_and_save_cutoffs(None)
# p2.plot_and_save_iplds(None) # IPLDS (the scatter graph)
# p2.plot_all_bdt() # Bandit characteristics
# p2.view_all_test_networks() # View top 5 networks of all tests
# p2.show_p_vals_krusk(True) # P-value for Kruskal
# p2.show_p_vals(False) # P-value for Mann-Whitney

test_names = ["Pendulum", "BipedalWalker", "BipedalWalkerHC", "LunarLander", "BanknoteAuth", "WineQuality", "MNIST", "MountainCar"]
bandit_names = ["Static-Conn", "N-Softmax t=0.05", "H-Softmax t=0.05", "N-Eps e=0.2", "H-Eps e=0.2", "H-TS", "N-TS",
                "Static-Node", "N-Softmax t=0.01", "H-Softmax t=0.01", "N-Eps e=0.01", "H-Eps e=0.01",
                "Static-Cross", "N-Softmax t=1.00", "H-Softmax t=1.00", "N-Eps e=0.8", "H-Eps e=0.8"]

def generate_salt(n):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(n))

def process_run(tst, bdt, run, show=False, plot=True):
    winner, stats, b_stats, config = p3.open_test(tst,bdt,run)

    generation, best_fitness, avg_fitness, median_fitness = p3.get_fitness_from_stats(stats)

    winner_gen = generation[-1]
    winner_fit = winner.fitness

    if not plot:
        return winner_gen, winner_fit
    
    processed_dir = f"processed/test_{tst}_{bdt}"
    
    t_name = test_names[tst]
    b_name = bandit_names[bdt]
    r_name = run

    p3.make_dir(tst, bdt)
    fitness_file = f"{processed_dir}/fitness_{tst}_{bdt}_{run}"
    p1.plot_fitness_from_list(generation, best_fitness, avg_fitness, t_name, b_name, r_name, save=fitness_file, show=show)

    species_file = f"{processed_dir}/species_{tst}_{bdt}_{run}"
    p1.plot_species_from_stats(stats, t_name, b_name, r_name, save=species_file, show=show)

    reward_file = f"{processed_dir}/reward_{tst}_{bdt}_{run}"
    count_file = f"{processed_dir}/count_{tst}_{bdt}_{run}"
    p3.plot_bdt_from_b_stats(b_stats,  t_name, b_name, r_name, savereward=reward_file, savecount=count_file, showreward=show, showcount=show)

    # Might need an additional "file_bdt_aggregate_from_b_stats" or something, and return it

    network_file = f"{processed_dir}/network_{tst}_{bdt}_{run}"
    p3.plot_top_n_networks_from_data(winner, stats, config, 4, save=network_file, show=show)
    
    return winner_gen, winner_fit

def process_test(tst, bdts=[0,1,2,3,4,5,6], runs=32, show=False, plot_indi=False):
    used_bdts = []
    gen_fits = [] # [[(g,f), (g,f), ...],[(g,f), (g,f), ...], ...]
    for bdt in bdts:
        used_bdts.append(bandit_names[bdt])
        gen_fits_bdt = []
        for run in range(runs):
            g,f = process_run(tst, bdt, run, show=False, plot=plot_indi)
            gen_fits_bdt.append((g,f))

        gen_fits.append(gen_fits_bdt)
    salt = generate_salt(5)
    gen_fit_file = f"processed/genfit_{tst}_{salt}"
    p3.plot_fitness_scatter_from_data(gen_fits, used_bdts, test_names[tst], save=gen_fit_file, show=show)
    
    box_file = f"processed/box_{tst}_{salt}"
    p3.plot_gen_fit_boxplot_from_data(gen_fits, used_bdts, test_names[tst], save=box_file, show=show)

    means_file = f"processed/means_{tst}_{salt}.csv"
    p2.show_mean_gens_fit_from_data(gen_fits, used_bdts, save=means_file, show=show)

    pairwise_file = f"processed/pairwise_{tst}_{salt}.csv"
    dominance_file = f"processed/dominance_{tst}_{salt}.csv"
    p2.show_pairwise_p_vals_mann_whitney_from_data(gen_fits, used_bdts, save=pairwise_file, save2=dominance_file, show=show)
    print(salt)
    
if __name__ == "__main__":
    complete_tests = [0,1,4,5,7]
    for tst in complete_tests:
        process_test(tst, bdts=range(17))
# process_test(0)
# process_test(1)
# process_test(2)
# process_test(3)
# process_test(4)
# process_test(5)
# process_test(6)