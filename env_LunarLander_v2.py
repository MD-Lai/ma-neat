import gym
import helper

env = gym.make("LunarLander-v2")
cfg = "cfg_LunarLander_v2"
name = "lunar"

# Evaluate genomes as required for NEAT
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        genome.fitness = 0

        net = helper.create_net(genome, config)

        total_reward = 0 
        observation = env.reset()
        
        for _ in range(1000):
            env.render()
            actions = net.activate(observation)
            
            # eng_lr = helper.scale(0,1,-1,1, eng_lr)
            a,_ = max(enumerate(actions), key=lambda i_s: i_s[1])
            
            observation,reward,done,info = env.step(a)

            total_reward += reward

            if done:
                break
    
        # Avg fitness over n runs
        genome.fitness = total_reward