import keras_quadopt as kqp
import pytest
import tensorflow as tf


def test1():
    with pytest.raises(Exception):
        kqp.aggregate_matrices()


def test2():
    with pytest.raises(Exception):
        kqp.aggregate_matrices([1])


def test3():
    with pytest.raises(Exception):
        kqp.aggregate_matrices([1], 1, [2])


def test4():
    simi1 = [[1., 2], [3, 4]]
    beta1 = 2.
    simi2 = [[5., 6], [7, 8]]
    beta2 = 0.5
    simi = kqp.aggregate_matrices(simi1, beta1, simi2, beta2)
    tf.debugging.assert_equal(simi, [[4.5, 7], [9.5, 12]])


def test5():
    simi1 = [[1, 2], [3, 4]]
    beta1 = 2
    simi2 = [[5, 6], [7, 8]]
    beta2 = 0.5
    simi = kqp.aggregate_matrices(simi1, beta1, simi2, beta2, dtype=tf.float16)
    assert simi.dtype == tf.float16
    tf.debugging.assert_equal(simi, tf.constant(
        [[4.5, 7], [9.5, 12]], dtype=tf.float16))
