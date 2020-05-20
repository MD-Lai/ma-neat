cfg = "cfg_Pendulum_v0"
name = "fast_test"

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = genome_id