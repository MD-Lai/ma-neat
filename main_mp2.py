import multiarm2
import bandit6
import informer

import neat
import visualize

import pickle
import sys
import traceback

import multiprocessing as mp
from datetime import datetime

# tests = [
#     env_Pendulum_v0,
#     env_BipedalWalker_v2,
#     env_BipedalWalkerHardcore_v2,
#     env_LunarLanderContinuous_v2,
#     cls_wine,
#     cls_banknote,
#     cls_MNIST
# ]


def run(ban, tst, test_id, gens=200):
    # Preferable to load locally, just to ensure no cross-talk or accidental "Sharing" of bandits by reference
    try:

        # There was a min/max in the env_ tests in env.observation_space.(high|low) to allow for auto scaling.
        if tst == 0:
            import env_Pendulum_v0 as t
        elif tst == 1:
            import env_BipedalWalker_v3 as t
        elif tst == 2: # fuck this one, no use 
            import env_BipedalWalkerHardcore_v3 as t
        elif tst == 3: # this one takes too long and for what 
            import env_LunarLander_v2 as t
        # Classifiers
        elif tst == 4:
            import cls_banknote as t
        elif tst == 5:
            import cls_wine as t
        elif tst == 6:
            import cls_MNIST as t
        elif tst == 7:
            import env_MountainCar_v0 as t
        elif tst == 8:
            import env_MountainCarContinuous_v0 as t
        elif tst == -1:
            import env_test as t
        else:
            print(f"No test defined for {tst}")
            exit()
        
        test = tst
        fit_func = t.eval_genomes
        cfg_file = t.cfg

        # if ban == 0:
        #     bandit = bandit4.RandomMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
        # elif ban == 1:
        #     bandit = bandit4.ProbMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
        # elif ban == 2:
        #     bandit = bandit4.HProbMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
        # elif ban == 3:
        #     bandit = bandit4.EpsMutator()
        # elif ban == 4:
        #     bandit = bandit4.HEpsMutator()
        # elif ban == 5:
        #     bandit = bandit4.TSMutator()
        # elif ban == 6:
        #     bandit = bandit4.HTSMutator()
        # else:
        #     print(f"No bandit defined for {ban}")
        #     exit()

        # Balanced parameters
        if ban == 0: # Connection preferred
            bandit = bandit6.Static([0.3, 0.1, 0.7, 0.5, 0.1])
        elif ban == 1: # Balanced greedy
            bandit = bandit6.N_Softmax(5, 0.05)
        elif ban == 2: # Balanced greedy
            bandit = bandit6.H_Softmax(5, 0.05)
        elif ban == 3: # Balanced greedy
            bandit = bandit6.N_Eps(5, 0.2)
        elif ban == 4: # Balanced greedy
            bandit = bandit6.H_Eps(5, 0.2)
        elif ban == 5: # No parameters req'd
            bandit = bandit6.H_TS(5)
        elif ban == 6: # No parameters req'd
            bandit = bandit6.N_TS(5)

        # Exploitation
        elif ban == 7: # Node preferred
            bandit = bandit6.Static([0.7, 0.5, 0.3, 0.1, 0.1])
        elif ban == 8: # Heavily Greedy
            bandit = bandit6.N_Softmax(5, 0.01)
        elif ban == 9:
            bandit = bandit6.H_Softmax(5, 0.01)
        elif ban == 10: # Heavily greedy
            bandit = bandit6.N_Eps(5, 0.01)
        elif ban == 11: # Heavily greedy
            bandit = bandit6.H_Eps(5, 0.01)
        

        # Exploration
        elif ban == 12: # Equal node and connections, crossover boosted
            bandit = bandit6.Static([0.7, 0.3, 0.7, 0.3, 0.5])
        elif ban == 13: # Exploration
            bandit = bandit6.N_Softmax(5, 1)
        elif ban == 14:
            bandit = bandit6.H_Softmax(5, 1)
        elif ban == 15: # Heavily greedy
            bandit = bandit6.N_Eps(5, 0.8)
        elif ban == 16: # Heavily greedy
            bandit = bandit6.H_Eps(5, 0.8)

        else:
            print(f"No bandit defined for {ban}")
            exit()

        b = bandit

        now = datetime.now()
        current_time = now.strftime("%D %H:%M:%S")

        print(f"Starting {current_time}: {test}_{b.simple_name()}_{test_id}")

        config = neat.Config(multiarm2.BanditGenome, multiarm2.BanditReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            cfg_file)

        p = neat.Population(config)

        p.reproduction.bandit_type(b)

        stats = neat.StatisticsReporter()
        b_stats = informer.BanditReporter()
        
        p.add_reporter(stats)
        p.add_reporter(b_stats)

        winner = p.run(fit_func, gens)
        
        test_prefix = f"results_{test}_{ban}_{test_id}"

        with open(f"{test_prefix}.pickle", 'wb') as gherkin:
            # info = (b.arm_history, b.reward_history, winner, stats, b_stats, config)
            info = (winner, stats, b_stats, config)
            pickle.dump(info, gherkin)
            # Saves:
            # (arm_history, reward_history, winner, neat_stats, bandit_stats, config)

        # shoouuuuuuuld be relatively easy to recreate/reload from loading the pickle
        # Done to reduce the total number of files generated
        
        #visualize.draw_net(config, winner, view=False, filename=test_prefix+"winner")
        #visualize.draw_net(config, stats.best_genome(), view=False, filename=test_prefix+"best")
        #visualize.plot_stats(stats, filename=test_prefix+"fitness")
        #visualize.plot_species(stats, filename=test_prefix+"speciation")
        
        now = datetime.now()
        current_time = now.strftime("%D %H:%M:%S")
        print(f"Complete {current_time}: {test}_{b.simple_name()}_{test_id}") 

    except Exception as e:
        print(f"Error in {tst}_{ban}_{test_id}")
        traceback.print_exc()
        print()
        raise e
        
if __name__ == "__main__":
    if len(sys.argv)-1 == 4:
        sc, tst, ban, cpu, sta = sys.argv
    else:
        print("Inappropriate parameters defined\ntest bandit n-cpu start")
        exit()
    
    tst = int(tst)
    ban = int(ban)
    cpu = int(cpu)
    sta = int(sta)

    # if tst == 0:
    #     import env_Pendulum_v0 as t
    # elif tst == 1:
    #     import env_BipedalWalker_v3 as t
    # elif tst == 2:
    #     import env_BipedalWalkerHardcore_v3 as t
    # elif tst == 3:
    #     import env_LunarLanderContinuous_v2 as t
    # # Classifiers
    # elif tst == 4:
    #     import cls_banknote as t
    # elif tst == 5:
    #     import cls_wine as t
    # elif tst == 6:
    #     import cls_MNIST as t
    # else:
    #     print(f"No test defined for {tst}")
    #     exit()
    
    # if ban == 0:
    #     bandit = bandit4.RandomMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
    # elif ban == 1:
    #     bandit = bandit4.PrMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
    # elif ban == 2:
    #     bandit = bandit4.EpsMutator()
    # elif ban == 3:
    #     bandit = bandit4.TSMutator()
    # else:
    #     print(f"No bandit defined for {ban}")
    #     exit()

    with mp.Pool(processes=cpu) as pool:
        for i in range(sta, sta+cpu):
            pool.apply_async(run, args=(ban, tst, i, 200))
        pool.close()
        pool.join()