import multiarm
import bandit4
import informer

import neat
import visualize

import pickle
from tqdm import tqdm

# import tests
import env_Pendulum_v0
import env_BipedalWalker_v2
import env_BipedalWalkerHardcore_v2
import env_LunarLanderContinuous_v2

def main():
    tests = [
        env_Pendulum_v0,
        env_BipedalWalker_v2,
        env_BipedalWalkerHardcore_v2,
        env_LunarLanderContinuous_v2
    ]

    for t in tqdm(tests):
        results = []
        for i in tqdm(range(50)):
            result = run(t.eval_genomes, t.cfg, test=f"{t.name}", test_id=f"{i}", gens=5)
            results.append(result)
        with open(f"{t.name}/results.pyc", 'wb') as gherkin:
            pickle.dump(results, gherkin)

def run(fit_func, cfg_file, test="", test_id="", gens=200):

    bandits = [
        bandit4.RandomMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True),
        bandit4.PrMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True),
        bandit4.EpsMutator(),
        bandit4.TSMutator()
    ]

    infos = []
    
    for b in bandits:
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

        test_prefix = f"{test}/{test_id}_{b.simple_name()}_"

        visualize.draw_net(config, winner, view=False, filename=test_prefix+"winner")
        visualize.draw_net(config, stats.best_genome(), view=False, filename=test_prefix+"best")
        visualize.plot_stats(stats, filename=test_prefix+"fitness")
        visualize.plot_species(stats, filename=test_prefix+"speciation")

        infos.append((winner, stats, b_stats, config))

    return infos

if __name__ == "__main__":
    main()