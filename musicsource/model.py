"""
Модель для генерации новых композиций

"""

import numpy as np  # Для генерации случайных чисел
import tensorflow as tf

from musicsource.moduleloader import ModuleLoader
from musicsource.keyboardcell import KeyboardCell
import musicsource.songstruct as music


class Model:
    """
    Базовый класс управления различными моделями
    """

    class TargetWeights:
        """ Политка установки весов
        """
        NONE = 'none'  # All weights equals (=1.0) (default)
        LINEAR = 'linear'  # The first outputs are less penalized than the last ones
        STEP = 'step'  # We start penalizing only after x steps (enco/deco behavior)

        def __init__(self, args):
            """
            """
            self.args = args

        def get_weight(self, i):
            """ 
            """
            if not self.args.target_weights or self.args.target_weights == Model.TargetWeights.NONE:
                return 1.0
            elif self.args.target_weights == Model.TargetWeights.LINEAR:
                return i / (self.args.sample_len - 1)  # Gradually increment the loss weight
            elif self.args.target_weights == Model.TargetWeights.STEP:
                raise NotImplementedError('Step target weight policy not implemented yet, please consider another policy')
            else:
                raise ValueError('Unknown chosen target weight policy: {}'.format(self.args.target_weights))

        @staticmethod
        def get_polics():
            """ Список политик по управлению весов
            """
            return [
                Model.TargetWeights.NONE,
                Model.TargetWeights.LINEAR,
                Model.TargetWeights.STEP
            ]

    class ScheduledSamplingPolicy:
        """ Контейнер для управления расписанием
        http://arxiv.org/abs/1506.03099 
        """
        NONE = 'none'  
        ALWAYS = 'always'  
        LINEAR = 'linear'  

        def __init__(self, args):
            self.sampling_policy_fct = None

            assert args.scheduled_smp
            assert len(args.scheduled_smp) > 0

            policy = args.scheduled_smp[0]
            if policy == Model.ScheduledSamplingPolicy.NONE:
                self.sampling_policy_fct = lambda step: 1.0
            elif policy == Model.ScheduledSamplingPolicy.ALWAYS:
                self.sampling_policy_fct = lambda step: 0.0
            elif policy == Model.ScheduledSamplingPolicy.LINEAR:
                if len(args.scheduled_smp) != 5:
                    raise ValueError('Not the right arguments for the sampling linear policy ({} instead of 4)'.format(len(args.scheduled_smp)-1))

                first_step = int(args.scheduled_smp[1])
                last_step = int(args.scheduled_smp[2])
                init_value = float(args.scheduled_smp[3])
                end_value = float(args.scheduled_smp[4])

                if (first_step >= last_step or
                   not (0.0 <= init_value <= 1.0) or
                   not (0.0 <= end_value <= 1.0)):
                    raise ValueError('Some schedule sampling parameters incorrect.')


                def linear_policy(step):
                    if step < first_step:
                        threshold = init_value
                    elif first_step <= step < last_step:
                        slope = (init_value-end_value)/(first_step-last_step)  # < 0 (потому что last_step>first_step и init_value>end_value)
                        threshold = slope*(step-first_step) + init_value
                    elif last_step <= step:
                        threshold = end_value
                    else:
                        raise RuntimeError('Invalid value for the sampling policy')  # Нв случай ошибки определния параметров
                    assert 0.0 <= threshold <= 1.0
                    return threshold

                self.sampling_policy_fct = linear_policy
            else:
                raise ValueError('Unknown chosen schedule sampling policy: {}'.format(policy))

        def get_prev_threshold(self, glob_step, i=0):
            """ Возвращает вероятность нажатия конфигурации нот, вычисленную на предыдущем шаге
            Args:
                glob_step (int) 
                i (int) 
            """
            return self.sampling_policy_fct(glob_step)

    def __init__(self, args):
        """
        Args:
            args: Параметры модели
        """
        print('Model creation...')

        self.args = args  

        # Заглушки
        self.use_prev = None
        self.current_lr = None
        self.inputs = None
        self.targets = None


        # Основные операторы
        self.opt_op = None  # Оптимизатор
        self.outputs = None  # Выходы нс
        self.final_state = None  # То что подается на вход

        # Другие параметры
        self.learning_rp = None
        self.loop_proc = None
        self.target_wp = None
        self.schedule_plc = None


        # НС
        self._build_network()

    def _build_network(self):
        """ Создание НС
        """
        input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()

        # Заглушки
        with tf.name_scope('placeholder_inputs'):
            self.inputs = [
                tf.placeholder(
                    tf.float32,  # -1.0/1.0 ? лучше для сигмоида..
                    [self.args.batch_size, input_dim],  
                    name='input')
                for _ in range(self.args.sample_len)
                ]
        with tf.name_scope('placeholder_targets'):
            self.targets = [
                tf.placeholder(
                    tf.int32,  # 0/1  # Int потому что sofmax
                    [self.args.batch_size,],  
                    name='target')
                for _ in range(self.args.sample_len)
                ]
        with tf.name_scope('placeholder_use_prev'):
            self.use_prev = [
                tf.placeholder(
                    tf.bool,
                    [],
                    name='use_prev')
                for _ in range(self.args.sample_len)
                ]

        # Определение НС
        self.loop_proc = ModuleLoader.loop_procs.build_module(self.args)
        def loop_rnn(prev, i):
            """ Функция цикла НС, связывающая выход одного слоя со входом другого.
            НС используется как для обучения, так и для генерации
            Args:
                prev: Предыдущая предсказанная конфигурация (шаг i-1)
                i: ID текущего шага (начинается с 1)
            Return:
                tf.Tensor
            """
            next_input = self.loop_proc(prev)

            # Во время обучения "запихиваем" правильных вход, во время генерации используем предыдущее значение как входное
            return tf.cond(self.use_prev[i], lambda: next_input, lambda: self.inputs[i])

        self.outputs, self.final_state = tf.nn.seq2seq.rnn_dec(
            decoder_inputs=self.inputs,
            initial_state=None, 
            cell=KeyboardCell(self.args),
            loop_function=loop_rnn
        )

        # Для обучения
        if not self.args.test:
            # Функция потерь

            # Мы используем стандарт = SoftMax где "метка" является относительной к положению предыдущей ноты.

            self.schedule_plc = Model.ScheduledSamplingPolicy(self.args)
            self.target_wp = Model.TargetWeights(self.args)
            self.learning_rp = ModuleLoader.learning_rate_policies.build_module(self.args)

            loss_fct = tf.nn.seq2seq.sequence_loss(
                self.outputs,
                self.targets,
                [tf.constant(self.target_wp.get_weight(i), shape=self.targets[0].get_shape()) for i in range(len(self.targets))],  # Веса
                #softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits, 
                average_across_timesteps=True,  
                average_across_batch=True  
            )
            tf.scalar_summary('training_loss', loss_fct)  

            self.current_lr = tf.placeholder(tf.float32, [])

            # Инициализация оптимизации функции потерь
            opt = tf.train.AdamOptimizer(
                learning_rate=self.current_lr,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )

            self.opt_op = opt.minimize(loss_fct)

    def step(self, batch, train_set=True, glob_step=-1, ret_output=False):
        """ Шаг обучения
        Args:
            batch (Batch)
            train_set (Bool)
            glob_step (int)
            ret_output (Bool)
        Return:
            Tuple[ops], dict 
        """

        feed_dict = {}
        ops = ()  
        batch.generate(target=False if self.args.test else True)

        # Заполнение заглушек
        if not self.args.test:  # Обучение
            if train_set:  # learning rate обновляется каждые x итераций 
                assert glob_step >= 0
                feed_dict[self.current_lr] = self.learning_rp.get_lr(glob_step)

            for i in range(self.args.sample_len):
                feed_dict[self.inputs[i]] = batch.inputs[i]
                feed_dict[self.targets[i]] = batch.targets[i]
                #if np.random.rand() >= self.schedule_plc.get_prev_threshold(glob_step)*self.target_wp.get_weight(i):
                if np.random.rand() >= self.schedule_plc.get_prev_threshold(glob_step):
                    feed_dict[self.use_prev[i]] = True
                else:
                    feed_dict[self.use_prev[i]] = False

            if train_set:
                ops += (self.opt_op,)
            if ret_output:
                ops += (self.outputs,)
        else:  # Генерация (batch_size == 1)
            for i in range(self.args.sample_len):
                if i < len(batch.inputs):
                    feed_dict[self.inputs[i]] = batch.inputs[i]
                    feed_dict[self.use_prev[i]] = False
                else:  
                    feed_dict[self.inputs[i]] = batch.inputs[0]  
                    feed_dict[self.use_prev[i]] = True  

            ops += (self.loop_proc.get_op(), self.outputs,)

        return ops, feed_dict
