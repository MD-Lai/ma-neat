import gym
import helper

env = gym.make("LunarLanderContinuous-v2")
cfg = "cfg_LunarLanderContinuous_v2"
name = "lunar"

n_runs = 5
# Evaluate genomes as required for NEAT
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        genome.fitness = 0

        net = helper.create_net(genome, config)

        total_reward = 0 
        for _ in range(n_runs):
            observation = env.reset()
            
            for _ in range(1000):
                eng_m, eng_lr = net.activate(observation)
                eng_lr = helper.scale(0,1,-1,1, eng_lr)

                observation,reward,done,info = env.step([eng_m, eng_lr])

                total_reward += reward

                if done:
                    break
        
        # Avg fitness over n runs
        genome.fitness = total_reward/n_runs