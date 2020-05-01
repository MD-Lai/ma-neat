import data
import helper
import random
import numpy as np

cfg = "cfg_banknote"
name = "bank"

data.LoadEnvironment("datasets/banknote/data_banknote_authentication.txt", "datasets/banknote/banknote.form")

x_test, y_test = data.values, data.targets

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)
        
        total_correct = 0
        # eval on the i'th bucket
        for x,y in zip(x_test,y_test):
            y_pred = net.activate(x)
            y_pred_i = max(range(len(y_pred)), key=lambda j: y_pred[j])
            y_test_i = max(range(len(y)), key=lambda j: y[j])

            # Get a measure of "how" correct it is
            correct = 1-((y[y_test_i] - y_pred[y_pred_i])) ** 2 if y_pred_i == y_test_i else 0
            avg_diff = sum([(y_p-y_t) ** 2 for y_p, y_t in zip(y_pred, y)])/len(y_pred)
            
            total_correct += correct - avg_diff
            
        genome.fitness = total_correct

