import tensorflow as tf


def aggregate_matrices(*args, dtype=tf.float32):
    # check inputs
    if len(args) % 2:
        raise Exception("An even number of input arguments expected")
    # copy the first matrix
    out = tf.constant(args[0], dtype=dtype) * tf.constant(args[1], dtype=dtype)
    # add the other matrices
    for i in range(2, len(args), 2):
        out += (tf.constant(args[i], dtype=dtype) * tf.constant(
            args[i + 1], dtype=dtype))
    # done
    return out
