import tensorflow as tf

def federated_mean():
        # TODO
        #
        # Add parameters, go through each local model that went through training. 
        # Each hidden layer weight matrix should be averaged with each local model, result returned as new global model weights.
        # So go through each hidden layer weight matrix, collect them from each local model, average them and store them.
        #
        
        # tf.math.reduce_mean(weights, axis=0) <- For each hidden layer for all local models that participated in training.
        return None