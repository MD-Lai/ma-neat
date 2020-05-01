import multiarm
import bandit4
import informer

import neat
import visualize

import pickle
import sys
import traceback

import multiprocessing as mp
from datetime import datetime

# tests = [
#     cls_wine,
#     cls_banknote,
#     env_Pendulum_v0,
#     env_BipedalWalker_v2,
#     env_BipedalWalkerHardcore_v2,
#     env_LunarLanderContinuous_v2,
#     cls_MNIST
# ]

def run(ban, tst, test_id, gens=200):
    # Preferable to load locally, just to ensure no cross-talk or accidental "Sharing" of bandits by reference
    try:

        if tst == 0:
            import env_Pendulum_v0 as t
        elif tst == 1:
            import env_BipedalWalker_v3 as t
        elif tst == 2:
            import env_BipedalWalkerHardcore_v3 as t
        elif tst == 3:
            import env_LunarLanderContinuous_v2 as t
        # Classifiers
        elif tst == 4:
            import cls_banknote as t
        elif tst == 5:
            import cls_wine as t
        else:
            print(f"No test defined for {tst}")
            exit()
        
        test = tst
        fit_func = t.eval_genomes
        cfg_file = t.cfg

        if ban == 0:
            bandit = bandit4.RandomMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
        elif ban == 1:
            bandit = bandit4.PrMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
        elif ban == 2:
            bandit = bandit4.EpsMutator()
        elif ban == 3:
            bandit = bandit4.TSMutator()
        else:
            print(f"No bandit defined for {ban}")
            exit()

        b = bandit

        now = datetime.now()
        current_time = now.strftime("%D %H:%M:%S")

        print(f"Starting {current_time}: {test}_{test_id}_{b.simple_name()}")

        config = neat.Config(multiarm.BanditGenome, multiarm.BanditReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            cfg_file)

        p = neat.Population(config)

        p.reproduction.bandit_type(b)

        stats = neat.StatisticsReporter()
        b_stats = informer.BanditReporter()
        
        p.add_reporter(stats)
        p.add_reporter(b_stats)

        winner = p.run(fit_func, gens)
        
        test_prefix = f"results_{test}_{test_id}_{b.simple_name()}_"

        # b.save(f"{test_prefix}bandit.pickle")

        with open(f"{test_prefix}bdt_win_stat_b_stat_cfg.pickle", 'wb') as gherkin:
            info = (b.arm_history, b.reward_history, winner, stats, b_stats, config)
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
        print(f"Complete {current_time}: {test}_{test_id}_{b.simple_name()}") 
    except Exception as e:
        print(f"Error in {tst}_{test_id}")
        traceback.print_exc()
        print()
        raise e

def fast_test(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = genome_id
        
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
    #     import cls_MNIST as t
    # elif tst == 5:
    #     import cls_banknote as t
    # elif tst == 6:
    #     import cls_wine as t
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
            # for t in tests:
            # pool.apply_async(run, args=(fast_test, t.cfg, bandit, "fast_test", f"{i}", 200))
            # print(i, hex(id(bandit)))
            # pool.apply_async(run, args=(t.eval_genomes, t.cfg, bandit, f"{t.name}", f"{i}", 200))
            pool.apply_async(run, args=(ban, tst, i, 200))
            # processes.append(mp.Process(target=run, args=(t.eval_genomes, t.cfg, f"{t.name}", f"{i}", 200)))
            # run(t.eval_genomes, t.cfg, test=f"{t.name}", test_id=f"{i}", gens=200)
        pool.close()
        pool.join()