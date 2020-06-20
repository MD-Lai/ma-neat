import gym
import helper

env = gym.make("MountainCarContinuous-v0")
cfg = "cfg_MountainCarContinuous_v0"
name = "mountain"

# Evaluate genomes as required for NEAT
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        genome.fitness = 0

        net = helper.create_net(genome, config)

        total_reward = 0 
        observation = env.reset()
        max_pos = 0
        max_speed = 0
        print(genome_id)
        for _ in range(200):
            # if genome_id > 200:
            # env.render()
            actions = net.activate(observation)
            
            # eng_lr = helper.scale(0,1,-1,1, eng_lr)
            
            observation,reward,done,info = env.step(actions)

            total_reward += reward

            max_pos = max(max_pos, abs(observation[0]))
            max_speed = max(max_speed, abs(observation[1]))

            if done:
                break
    
        # Avg fitness over n runs
        genome.fitness = total_reward + max_pos + max_speed
        print(genome.fitness)