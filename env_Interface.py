import helper

cfg = "config_file_name_for_this_test"
name = "short_identifier_for_this_test"

# Evaluate genomes as required for NEAT
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        total_reward = 0

        net = helper.create_net(genome, config)

        # do something and assign reward to genome.fitness
