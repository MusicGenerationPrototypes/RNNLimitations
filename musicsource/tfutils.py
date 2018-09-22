"""
Вспомогательные функции для НС (взято с tf)
"""

import tensorflow as tf


def single_layer_perceptron(shape, scope_name):
    """ Single layer perceptron
    Project X on the output dimension
    Args:
        shape: a tuple (input dim, output dim)
        scope_name (str): encapsulate variables
    Return:
        tf.Ops: The projection operator (see project_fct())
    """
    assert len(shape) == 2

    # Projection on the keyboard
    with tf.variable_scope('weights_' + scope_name):
        W = tf.get_variable(
            'weights',
            shape,
            initializer=tf.truncated_normal_initializer() 
        )
        b = tf.get_variable(
            'bias',
            shape[1],
            initializer=tf.constant_initializer()
        )

    def project_fct(X):
        """ Project the output of the decoder into the note space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(scope_name):
            return tf.matmul(X, W) + b

    return project_fct


def get_rnn_cell(args, scope_name):
    """ Return RNN cell, constructed from the parameters
    Args:
        args: the rnn parameters
        scope_name (str): encapsulate variables
    Return:
        tf.RNNCell: a cell
    """
    with tf.variable_scope('weights_' + scope_name):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hidden_size)
        #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=1.0) 
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * args.num_layers, state_is_tuple=True)
    return rnn_cell
