import sys

import neat
import visualize

import multiarm
import bandit3
import custom_reporter
import helper

import gym

import random
import math

import pickle
from tqdm import tqdm

env = None
total_repeats = 50

cartpole = "CartPole-v1"
bipedal = "BipedalWalker-v2"
bipedal_hard = "BipedalWalkerHardcore-v2"
lunar = "LundarLanderContinuous-v2"

# TODO change these to the file names
cifar = "Cifar10"
mnist = "MNIST"
wine = "Wine"
notes = "banknotes"

control_suite = [cartpole, bipedal, lunar]
classify_suite = [cifar, mnist, wine, notes]

def run_tests():

    cfg = "config-feedforward-bandit-pendulum"
    gens = 200
    bandit = bandit3.PRMutator(rates=[0.2, 0.1, 0.8, 0.5, 0.2, 0.9], lr=0.01)

    for i in tqdm(range(total_repeats)):
        result = run(cfg, gens, eval_CartPole_v1, bandit)
        

def run(cfg, gens, fit, bandit, std_out=False):
    config = neat.Config(multiarm.BanditGenome, multiarm.BanditReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        cfg)
    
    p = neat.Population(config)

    p.reproduction.bandit_type(bandit)

    if std_out:
        p.add_reporter(neat.StdOutReporter(True))
    
    stats = neat.StatisticsReporter()
    bandit_stats = custom_reporter.BanditReporter()

    p.add_reporter(stats)
    p.add_reporter(bandit_stats)

    winner = p.run(fit, gens)

    return winner, stats, bandit_stats, config

def get_env(env_name):
    global env

    # Control
    if env_name == cartpole:
        env = gym.make(cartpole)
        return eval_CartPole_v1
    elif env_name == bipedal:
        env = gym.make(bipedal)
        return eval_BipedalWalker_v2
    elif env_name == lunar:
        env = gym.make(lunar)
        return eval_LundarLander_v2
    
    # Classification
    # TODO make a classification "environment"
    elif env_name == cifar:
        env = None 
    elif env_name == mnist:
        env = None
    elif env_name == wine:
        env = None
    elif env_name == notes:
        env = None

    else:
        print("No such environment: {}".format(env_name))
        env = None

def create_net(genome, config):
    # Just to cut lines from every function
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        return neat.nn.RecurrentNetwork.create(genome, config)

def eval_CartPole_v1(genomes, config):

    for genome_id, genome in genomes:

        observation = env.reset()

        if config.genome_config.feed_forward:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        
        genome.fitness = 0

        for _ in range(100): # 2 seconds at 50fps 
            # Adjust observations
            
            # Observation: 
            #     Type: Box(4)
            #     Num	Observation                 Min         Max
            #     0	    Cart Position             -4.8            4.8
            #     1	    Cart Velocity             -Inf            Inf
            #     2	    Pole Angle                 -24 deg        24 deg
            #     3	    Pole Velocity At Tip      -Inf            Inf

            pos, vel, ang, tip = observation
            # Not sure how to scale speed, initial runs show that it doesn't get very high (and it shouldn't anyway)
            pos = helper.scale(-4.8, 4.8, -1, 1, pos)
            vel = vel 
            ang = helper.scale(-24, 24, -1, 1, ang)
            tip = tip

            re_obv = [pos, vel, ang, tip]

            # Actions:
            #     Type: Discrete(2)
            #     Num	Action
            #     0	    Push cart to the left
            #     1	    Push cart to the right
    
            action = net.activate(re_obv)[0]
            action = 0 if action < 0.5 else 1

            observation, reward, done, info = env.step(action)
            genome.fitness += reward

            if done:
                break

def eval_BipedalWalker_v2():
    # 250 iters 5 seconds at 50fps
    # 24 inputs, 14 features + 10 lidar
    # 4 outputs, -1 to 1 for each joint
    pass

def eval_BipedalWalkerHardcore_v2(genome, config):
    # functionally the same as bipedalwalker
    pass

def eval_LundarLander_v2():
    # 200 iters 4 seconds at 50fps
    # 8 inputs, 
    pass

def eval_CIFAR_10():
    pass

def eval_MNIST():
    pass

def eval_Wine_Quality():
    pass

def eval_Banknote():
    pass

# run_tests()