import helper

import keras
from keras.datasets import mnist

cfg = "cfg_MNIST"
name = "MNIST"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = helper.create_net(genome, config)

        sse = 0

        for x,y in zip(x_test, y_test):
            y_pred = net.activate(x)
            sse += sum([(p-c)**2 for p,c in zip(y_pred, y)])
            
        genome.fitness = -sse