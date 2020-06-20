import gym
import helper

env = gym.make("MountainCar-v0")
cfg = "cfg_MountainCar_v0"
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
        min_speed = 0
        for _ in range(200):
            position_scaled = helper.scale(-1.2, 0.6, 0, 1, observation[0])
            velocity_scaled = helper.scale(-0.07, 0.07, -1, 1, observation[1])
            # if genome_id > 200:
            #     env.render()
                # print(position_scaled, velocity_scaled)
            actions = net.activate([position_scaled, velocity_scaled])
            action,_ = max(enumerate(actions), key=lambda i_s: i_s[1])
            # eng_lr = helper.scale(0,1,-1,1, eng_lr)
            
            observation,reward,done,info = env.step(action)

            total_reward += reward

            max_pos = max(max_pos, observation[0])
            max_speed = max(max_speed, abs(observation[1]))
            min_speed = min(min_speed, abs(observation[1]))

            if done:
                break
    
        # Avg fitness over n runs
        genome.fitness = total_reward + max_pos + max_speed + (max_speed-min_speed)

