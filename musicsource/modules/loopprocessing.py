"""
"""

import tensorflow as tf


class LoopProcessing:
    """ 
    """
    def __init__(self, args):
        pass

    def __call__(self, prev_output):
        """ 
        """
        raise NotImplementedError('Abstract Class')

    def get_op(self):
        """ 
        """
        return ()  


class SampleSoftmax(LoopProcessing):
    """ 
    """
    @staticmethod
    def get_module_id():
        return 'sample_softmax'

    def __init__(self, args, *args_module):

        self.temperature = args.temperature  # Control the sampling (more or less concervative predictions)
        self.chosen_labels = []  # Keep track of the chosen labels (to reconstruct the chosen song)

    def __call__(self, prev_output):
        """ 

        """
        # prev_output size: [batch_size, nb_labels]
        nb_labels = prev_output.get_shape().as_list()[-1]

        if False:
            #label_draws = tf.argmax(prev_output, 1)
            label_draws = tf.multinomial(tf.log(prev_output), 1)  # Draw 1 sample from the distribution
            label_draws = tf.squeeze(label_draws, [1])
            self.chosen_labels.append(label_draws)
            next_input = tf.one_hot(label_draws, nb_labels)
            return next_input
        # Could use the Gumbel-Max trick to sample from a softmax distribution ?

        soft_values = tf.exp(tf.div(prev_output, self.temperature))  # Pi = exp(pi/t)
        # soft_values size: [batch_size, nb_labels]

        normalisation_coeff = tf.expand_dims(tf.reduce_sum(soft_values, 1), -1)
        # normalisation_coeff size: [batch_size, 1]
        probs = tf.div(soft_values, normalisation_coeff + 1e-8)  # = Pi / sum(Pk)
        # probs size: [batch_size, nb_labels]
        label_draws = tf.multinomial(tf.log(probs), 1)  # Draw 1 sample from the log-probability distribution
        # probs label_draws: [batch_size, 1]
        label_draws = tf.squeeze(label_draws, [1])
        # label_draws size: [batch_size,]
        self.chosen_labels.append(label_draws)
        next_input = tf.one_hot(label_draws, nb_labels)  # Reencode the next input vector
        # next_input size: [batch_size, nb_labels]
        return next_input

    def get_op(self):
        """ 
        """
        return self.chosen_labels


class ActivateScale(LoopProcessing):
    """ [-1, 1]
    """
    @staticmethod
    def get_module_id():
        return 'activate_and_scale'

    def __init__(self, args):
        pass

    def __call__(X):
        """ [-1, 1]
        Use sigmoid activation
        Args:
            X (tf.Tensor): the input
        Return:
            tf.Ops: the activate_and_scale operator
        """
        with tf.name_scope('activate_and_scale'):
            return tf.sub(tf.mul(2.0, tf.nn.sigmoid(X)), 1.0)  #
