"""
Основной блок, предсказывающий конфигурацию следующей нажатой последовательности нот.
"""

import collections
import tensorflow as tf

from musicsource.moduleloader import ModuleLoader
import musicsource.songstruct as music


class KeyboardCell(tf.nn.rnn_cell.RNNCell):
    """ encoder/decoder network
    """

    def __init__(self, args):
        self.args = args
        self.is_init = False

        # Get the chosen enco/deco
        self.encoder = ModuleLoader.enco_cells.build_module(self.args)
        self.decoder = ModuleLoader.deco_cells.build_module(self.args)

    @property
    def state_size(self):
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self):
        raise NotImplementedError('Abstract method')

    def __call__(self, prev_keyboard, prev_state, scope=None):
        """ Запуск блока на t-м шаге
        Аргументы:
            prev_keyboard: keyboard configuration for the step t-1 (Ground truth or previous step)
            prev_state: a tuple (prev_state_enco, prev_state_deco)
            scope: TensorFlow scope
        Возвращает:
            Tuple: the keyboard configuration and the enco and deco states
        """

        if not self.is_init:
            with tf.variable_scope('weights_keyboard_cell'):
                self.encoder.build()
                self.decoder.build()

                prev_state = self.encoder.init_state(), self.decoder.init_state()
                self.is_init = True

        # Encoder/decoder network
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('Encoder'):
                next_state_enco = self.encoder.get_cell(prev_keyboard, prev_state)
            with tf.variable_scope('Decoder'):  # Перезапуск и обновление gate.
                next_keyboard, next_state_deco = self.decoder.get_cell(prev_keyboard, (next_state_enco, prev_state[1]))
        return next_keyboard, (next_state_enco, next_state_deco)
