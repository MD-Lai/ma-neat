"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
import multiarm
import custom_reporter
from random import random
import gym
import bandit4

from tqdm import tqdm # The progress bar

# e.g. use of tqdm, wrap whatever iterable in tqdm. 
# Most relevant parameters is desc, to prefix a descriptor to the progress bar
# for i in tqdm(range(500), desc="Test 1: "):
#     pass # do stuff

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# pendulum
env = gym.make("Pendulum-v0")
def eval_genomes(genomes, config):
    observation = env.reset()
    
    for genome_id, genome in genomes:
        observation = env.reset()
        if config.genome_config.feed_forward:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        genome.fitness = 0

        for _ in range(200):
            # print(observation)
            action = (net.activate(observation)[0]-0.5) * 4
            # print(action)
            observation, reward, done, info = env.step([action])
            genome.fitness += reward

            if done:
                break


# def eval_genomes(genomes, config):
#     for genome_id, genome in genomes:
#         fitness = 0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for xi, xo in zip(xor_inputs, xor_outputs):
#             output = net.activate(xi)
#             fitness += 1 - (output[0] - xo[0]) ** 2
        
#         genome.fitness = fitness / len(xor_inputs)

def true_xor(a_in,b_in):
    if a_in > 0.5:
        a = 1
    else:
        a = 0
    
    if b_in > 0.5:
        b = 1
    else:
        b = 0

    if not a == b:
        return 1
    return 0

def run(config_file):
    # Load configuration.
    config = neat.Config(multiarm.BanditGenome, multiarm.BanditReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add bandit here
    # bandit = bandit3.RandomMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9])
    # bandit = bandit4.PrMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], single=True)
    # bandit = bandit4.EpsMutator(plays=[1])
    bandit = bandit4.TSMutator()
    p.reproduction.bandit_type(bandit)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    br = custom_reporter.BanditPrintReporter()
    p.add_reporter(br)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 200 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # for _ in range(10):
    #     x,y = (random(), random())
    #     output = winner_net.activate((x,y))
    #     print("input {!r}, expected output {!r}, got {!r}".format((x,y), (true_xor(x,y),), output))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, view=True)
    visualize.draw_net(config, stats.best_genome(), view=True, filename="best_genome")
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-bandit-pendulum')
    run(config_path)