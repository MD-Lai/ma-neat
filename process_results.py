import sys
import pickle
import visualize

_, pickle_file = sys.argv

with open(pickle_file, 'rb') as pickle_in:
    b_arm, b_reward, winner, stats, b_stats, config = pickle.load(pickle_in)

print(f"b_arm:{sys.getsizeof(b_arm)}")
print(f"b_reward:{sys.getsizeof(b_reward)}")
print(f"winner:{sys.getsizeof(winner)}")
print(f"stats:{sys.getsizeof(stats)}")
print(f"b_stats:{sys.getsizeof(b_stats)}")
print(f"config:{sys.getsizeof(config)}")

# test_prefix = "pickle_test_"
# visualize.draw_net(config, winner, view=False, filename=test_prefix+"winner")
# visualize.draw_net(config, stats.best_genome(), view=False, filename=test_prefix+"best")
# visualize.plot_stats(stats, filename=test_prefix+"fitness")
# visualize.plot_species(stats, filename=test_prefix+"speciation")   


#visualize.draw_net(config, winner, view=False, filename=test_prefix+"winner")
#visualize.draw_net(config, stats.best_genome(), view=False, filename=test_prefix+"best")
#visualize.plot_stats(stats, filename=test_prefix+"fitness")
#visualize.plot_species(stats, filename=test_prefix+"speciation")