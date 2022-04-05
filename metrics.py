import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

    
def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Sigmoid cross-entropy loss with masking"""
    # loss has the same shape as logits: 1 loss per class and per sample in the batch
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_sum(loss, axis=1)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def multi_label_hot(prediction, threshold=0.5):
    """
    Examples:
        prediction = tf.sigmoid(logits)
        one_hot_prediction = multi_label_hot(prediction)
    """
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater_equal(prediction, threshold), tf.int64)


def f1_score(y_true, y_pred, mask, epsilon=1e-8):
    f1s = [0, 0, 0]
    
    y_true = tf.cast(tf.boolean_mask(y_true, mask, axis=0), tf.float64)
    y_pred = tf.cast(tf.boolean_mask(y_pred, mask, axis=0), tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.cast(tf.count_nonzero(y_pred * y_true, axis=axis), tf.float64)
        FP = tf.cast(tf.count_nonzero(y_pred * (y_true - 1), axis=axis), tf.float64)
        FN = tf.cast(tf.count_nonzero((y_pred - 1) * y_true, axis=axis), tf.float64)

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights + epsilon)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted, TP, FP, FN
