import math
import neat

def scale(in_low, in_high, out_low, out_high, val):
    return ((out_high - out_low) * (val - in_low)) /(in_high - in_low) + out_low

def create_net(genome,config):
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        return neat.nn.RecurrentNetwork.create(genome, config)