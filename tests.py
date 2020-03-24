import neat

import gym

# cart pole
env = gym.make("CartPole-v1")
def eval_genomes(genomes, config):

    observation = env.reset()
    
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        for _ in range(500):
            action = 1 if net.activate(observation)[0] > 0.5 else 0

            observation, reward, done, info = env.step(action)
            genome.fitness += reward


        

# pendulum
env = gym.make("Pendulum-v0")
def eval_genomes(genomes, config):
    observation = env.reset()
    
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        for _ in range(200):
            action = net.activate(observation)[0] * 2

            observation, reward, done, info = env.step([action])
            genome.fitness += reward


    
# bipedal walker
env = gym.make("BipedalWalker-v2")
def eval_genomes(genomes, config):
    observation = env.reset()
    
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        for _ in range(100):
            action = net.activate(observation)[0]

            observation, reward, done, info = env.step([action])
            genome.fitness += reward