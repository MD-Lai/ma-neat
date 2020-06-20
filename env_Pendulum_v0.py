import neat
import gym
import helper

env = gym.make("Pendulum-v0")
cfg = "cfg_Pendulum_v0"
name = "pendulum"

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)
        total_reward = 0

        observation = env.reset()

        for _ in range(200):
            cos_th, sin_th, th_dot = observation
            th_dot = helper.scale(-8, 8, -1, 1, th_dot)
            action = (net.activate([cos_th, sin_th, th_dot])[0])
            observation, reward, done, info = env.step([action])
            total_reward += reward

            if done:
                break

        genome.fitness = total_reward