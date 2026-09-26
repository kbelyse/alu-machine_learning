#!/usr/bin/env python3
"""
Function def create_RMSProp_op(loss, alpha, beta2, epsilon)
that creates the training operation for a neural network in tensorflow
using the RMSProp optimization algorithm
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm

    Args:
        loss: the loss of the network
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero

    Returns:
        the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)
