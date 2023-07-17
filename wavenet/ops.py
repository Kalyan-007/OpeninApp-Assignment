from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.compat.v1.name_scope('time_to_batch'):
        shape = tf.compat.v1.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.compat.v1.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.compat.v1.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.compat.v1.transpose(reshaped, perm=[1, 0, 2])
        return tf.compat.v1.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.compat.v1.name_scope('batch_to_time'):
        shape = tf.compat.v1.shape(value)
        prepared = tf.compat.v1.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.compat.v1.transpose(prepared, perm=[1, 0, 2])
        return tf.compat.v1.reshape(transposed,
                          [tf.compat.v1.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.compat.v1.name_scope(name):
        filter_width = tf.compat.v1.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.compat.v1.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.compat.v1.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.compat.v1.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.compat.v1.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.compat.v1.name_scope('encode'):
        mu = tf.compat.v1.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.compat.v1.minimum(tf.compat.v1.abs(audio), 1.0)
        magnitude = tf.compat.v1.log1p(mu * safe_audio_abs) / tf.compat.v1.log1p(mu)
        signal = tf.compat.v1.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.compat.v1.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.compat.v1.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.compat.v1.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.compat.v1.sign(signal) * magnitude
