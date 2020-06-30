import gym
import helper

env = gym.make("BipedalWalker-v2")
cfg = "cfg_BipedalWalker_v3"
name = "bipedal"

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)
        # 1600 steps to "solve" as per environment requests (most are expected to fail anyway)
        # 24 inputs, 14 features + 10 lidar
        # 4 outputs, -1 to 1 for each joint
        total_reward = 0
        observation = env.reset()
        stopped_time = 0 
        for _ in range(800):
            env.render()
            action = net.activate(observation)
            observation,reward,done,info = env.step(action)

            if abs(observation[2]) < 0.001:
                stopped_time += 1
            if stopped_time >= 50:
                reward = -100
                done = True

            total_reward += reward
            if done:
                break

        genome.fitness = total_reward