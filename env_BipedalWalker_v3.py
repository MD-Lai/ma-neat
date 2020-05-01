import gym
import helper

env = gym.make("BipedalWalker-v2")
cfg = "cfg_BipedalWalker_v2"
name = "bipedal"

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)
        # 1600 steps to "solve" as per environment requests (most are expected to fail anyway)
        # 24 inputs, 14 features + 10 lidar
        # 4 outputs, -1 to 1 for each joint
        total_reward = 0
        observation = env.reset()
        for _ in range(800):
            action = net.activate(observation) 
            # These are all standard sigmoid outputs = [0,1]
            # Adjust to [-1,1]
            for i,a in enumerate(action):
                action[i] = helper.scale(0,1,-1,1, a)
            observation,reward,done,info = env.step(action)
            total_reward += reward
            if done:
                break

        genome.fitness = total_reward