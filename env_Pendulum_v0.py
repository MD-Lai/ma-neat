import neat
import gym
import helper

env = gym.make("Pendulum-v0")
cfg = "cfg_Pendulum_v0"
name = "pendulum"

n_runs = 5

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)
        total_reward = 0

        for _ in range(n_runs):
            observation = env.reset()

            for _ in range(200):
                cos_th, sin_th, th_dot = observation
                th_dot = helper.scale(-8, 8, -1, 1, th_dot)
                action = (net.activate([cos_th, sin_th, th_dot])[0]-0.5) * 4
                observation, reward, done, info = env.step([action])
                total_reward += reward

                if done:
                    break

        genome.fitness = total_reward/n_runs