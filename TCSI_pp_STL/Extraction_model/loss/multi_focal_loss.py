from keras import backend as K
import tensorflow as tf
import numpy as np


def KerasFocalLoss(target, input, Num):  # Num为样本数量

    gamma = 2.
    input = tf.cast(input, tf.float32)

    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.compat.v1.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    W = 1 / np.log(Num)
    W = tf.cast(W, tf.float32)
    we_loss = tf.compat.v1.matmul(loss, W)

    return K.mean(K.sum(we_loss, axis=1))
