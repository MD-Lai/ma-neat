import process_results as p1
import process_results_2 as p2

p1.plot_and_save_fitnesses(range(0,7), range(0,7), None) # Fitnesses
p2.plot_and_save_iplds(None) # IPLDS (the scatter graph)
p2.plot_all_bdt() # Bandit characteristics
p2.view_all_test_networks() # View top 5 networks of all tests
p2.show_p_vals_krusk(True) # P-value for Kruskal
p2.show_p_vals(False) # P-value for Mann-Whitney