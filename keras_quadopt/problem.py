import tensorflow as tf
from typing import List, Optional


def get_weights(c: List[float],
                Q: List[List[float]],
                lam: Optional[float] = 0.5,
                dtype=tf.float32,
                maxiter: Optional[int] = 500,
                ftol: Optional[float] = 1e-06,
                patience: int = 50
                ):
    """Maximize "total goodness" and minimize "total similarity" """
    # error checking
    if lam < 0:
        raise Exception(f"the preference lambda='{lam}' must be positive")

    # how big is `N`
    n_examples = len(c)

    # We can multipy `lam*Q` beforehand and save compute time!
    lamQ = tf.constant(Q) * lam  # (n,n)
    cin = tf.constant(c)  # (n,)

    # initial values
    w0 = tf.ones(shape=(n_examples,)) / n_examples
    # trainable params
    w = tf.Variable(
        initial_value=w0, trainable=True, dtype=dtype)

    # loss function with regularization
    def custom_loss():
        # norm to 1
        v = w / tf.maximum(1e-8, tf.reduce_sum(w))
        # quadratic problem
        loss = -tf.tensordot(cin, v, axes=1)
        loss += tf.sqrt(tf.tensordot(tf.tensordot(lamQ, v, axes=1), v, axes=1))
        # regularization: sum_i w_i = 1
        loss += tf.pow(1. - tf.reduce_sum(w), 2)
        # regularization: w_i >= 0
        loss += tf.reduce_sum(-tf.minimum(w, 0.0))
        return loss

    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=.9, beta_2=.999,
        epsilon=1e-7, amsgrad=True)  # 3e-4

    # start loop
    fbest = custom_loss()
    wbest = w.numpy()
    wait = 0
    for i in range(maxiter):
        optimizer.minimize(loss=custom_loss, var_list=[w])
        f = custom_loss()
        if fbest > (f + ftol):
            fbest = f
            wbest = w.numpy()
            wait = 0
        else:
            wait += 1
            if wait > patience:
                break

    # done
    return wbest, fbest
