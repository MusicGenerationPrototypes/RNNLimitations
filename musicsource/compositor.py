
"""
Класс-композитор, запускает процесс обучения или генерации.
Код частично адаптирован из https://github.com/Conchylicultor/MusicGenerator
"""

import argparse  # Парсинг параметров командной строки
import configparser  # Сохранение параметров модели
import datetime  # Дата и время
import os  # Управление файлами
from tqdm import tqdm  # Шкала загрузки
import tensorflow as tf
import gc  # Сборщик мусора перед каждой эпохой

from musicsource.moduleloader import ModuleLoader
from musicsource.musicdata import MusicData
from musicsource.midiconnect import midiconnect
from musicsource.imgconnector import ImgConnector
from musicsource.model import Model


class compositor:
    """
    Основной класс, запускающий процесс обучения или генерации
    """

    class TestMode:
        """ Режимы работы
        """
        ALL = 'all'  # Генерация композиций
        DAEMON = 'daemon'  #В планах
        INTERACTIVE = 'interactive'  #В планах

        @staticmethod
        def get_test_modes():
            """ Возвращает режим работы
            """
            return [compositor.TestMode.ALL, compositor.TestMode.DAEMON, compositor.TestMode.INTERACTIVE]

    def __init__(self):
        """
        """
        # Параметры датасета и модели
        self.args = None

        # Задание специальных объектов
        self.music_data = None  # Dataset
        self.model = None  # базовая модель

        # tf
        self.writer = None
        self.writer_test = None
        self.saver = None
        self.model_dir = ''  # путь сохранения
        self.glob_step = 0

        # Сессия
        self.sess = None

        # Имена файлов и константы
        self.MODEL_DIR_BASE = 'save\model' #\ вместо /
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.5'

        self.TRAINING_VISUALIZATION_STEP = 1000  
        self.TRAINING_VISUALIZATION_DIR = 'progression'
        self.TESTING_VISUALIZATION_DIR = 'midi'  

    @staticmethod
    def _parse_args(args):
        """
        Парсинг параметров при запуске
        Args:
            args (list<str>): Список аргументов на парсинг. При отсутсвии, подаются значения по умолчанию.
        """

        parser = argparse.ArgumentParser()

        # Глобальные опции
        global_args = parser.add_argument_group('Global options')
        global_args.add_argument('--test', nargs='?', choices=compositor.TestMode.get_test_modes(), const=compositor.TestMode.ALL, default=None,
                                 help='Генерация композиций')
        global_args.add_argument('--reset', action='store_true', help='Игнорировать предыдущую модель, представленную в папке с моделями (Осторожно! Предыдущая модель будет удалена со всеми вложенными папками!)')
        global_args.add_argument('--keep_all', action='store_true', help='Если выбрано, все сохраненные модели будут оставлены (Осторожно! Убедитесь в том, что есть достаточно места на жестком диске либо увеличьте параметр save_every)')
        global_args.add_argument('--model_tag', type=str, default=None, help='Тэг для различия, какую модель хранить/загружать')
        global_args.add_argument('--sample_len', type=int, default=40, help='Количество (временных) шагов тренеровочной последовательности, длина генерируемой последовательности')  # Warning: the unit is defined by the MusicData.MAXIMUM_SONG_RESOLUTION parameter
        global_args.add_argument('--root_dir', type=str, default=None, help='Папка, в которой следует искать модели и данные')
        global_args.add_argument('--device', type=str, default=None, help='\'gpu\' или \'cpu\' (Осторожно! Убедитесь в том, что у вас достаточно оперативной памяти), Позволяет выбрать на какой модели памяти использовать сеть')
        global_args.add_argument('--temperature', type=float, default=1.0, help='Используется при тестировании, контролирует выборку выходных данных')

        # Опции датасета
        dataset_args = parser.add_argument_group('Dataset options')
        dataset_args.add_argument('--dataset_tag', type=str, default='musicdataset', help='Выбор данных для обучения')
        dataset_args.add_argument('--create_dataset', action='store_true', help='Генерация датасета из папки')
        dataset_args.add_argument('--play_dataset', type=int, nargs='?', const=10, default=None,  help='Воспроизведение некоторых частей')
        dataset_args.add_argument('--ratio_dataset', type=float, default=0.9, help='Соотношение песен для обучения\тестирования. Задается в начале и не может быть изменено')
        ModuleLoader.batch_builders.add_argparse(dataset_args, 'Контроль представления результатов')

        # Опции НС
        nn_args = parser.add_argument_group('Network options', 'architecture related option')
        ModuleLoader.enco_cells.add_argparse(nn_args, 'Encoder cell used.')
        ModuleLoader.deco_cells.add_argparse(nn_args, 'Decoder cell used.')
        nn_args.add_argument('--hidden_size', type=int, default=512, help='Размер слоя НС')
        nn_args.add_argument('--num_layers', type=int, default=2, help='Количество рекуррентных слоев НС')
        nn_args.add_argument('--scheduled_smp', type=str, nargs='+', default=[Model.ScheduledSamplingPolicy.NONE], help='Определение расписания. Показывает параметры выбранной политики')
        nn_args.add_argument('--target_weights', nargs='?', choices=Model.TargetWeights.get_polics(), default=Model.TargetWeights.LINEAR,
                             help='вклад за каждый шаг обучения')
        ModuleLoader.loop_procs.add_argparse(nn_args, 'Преобразования для пременения на каждом выходе')

        # Опции обучения
        training_args = parser.add_argument_group('Training options')
        training_args.add_argument('--num_epochs', type=int, default=0, help='Максимальное количество эпох для запуска (0 для бесконечности)')
        training_args.add_argument('--save_every', type=int, default=1000, help='количество мини-батчей перед каждым промежуточным сохранением')
        training_args.add_argument('--batch_size', type=int, default=64, help='количество батчей')
        ModuleLoader.learning_rate_policies.add_argparse(training_args, 'Коэффициент скорости обучения (Learning rate)')
        training_args.add_argument('--testing_curve', type=int, default=10, help='Запись тестирования каждую х-ю итерацию')

        return parser.parse_args(args)

    def main(self, args=None):
        """
        Запуск обучения
        """
        print('Добро пожаловать в music genetaror!')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # Общие установки

        tf.logging.set_verbosity(tf.logging.INFO)  

        ModuleLoader.register_all()  # Загрузка модулей
        self.args = self._parse_args(args)
        if not self.args.root_dir:
            self.args.root_dir = os.getcwd()  

        self._restore_params()  # обновление директорий и параметров
        self._print_params()

        self.music_data = MusicData(self.args)
        if self.args.create_dataset:
            print('Датасет создан! Можно начать обучение')
            return  # No need to go further

        with tf.device(self._get_device()):
            self.model = Model(self.args)

        # Сохранение
        self.writer = tf.train.SummaryWriter(os.path.join(self.model_dir, 'train'))
        self.writer_test = tf.train.SummaryWriter(os.path.join(self.model_dir, 'test'))
        self.saver = tf.train.Saver(max_to_keep=200)  #


        # Сессия

        self.sess = tf.session()

        print('Инициализация переменных...')
        self.sess.run(tf.initialize_all_variables())

        
        self._restore_previous_model(self.sess)

        if self.args.test:
            if self.args.test == compositor.TestMode.ALL:
                self._main_test()
            elif self.args.test == compositor.TestMode.DAEMON:
                print('запуск... ')
                raise NotImplementedError('Такого режима не существует')  
            else:
                raise RuntimeError('Неизвестный режим: {}'.format(self.args.test)) 
        else:
            self._main_train()

        if self.args.test != compositor.TestMode.DAEMON:
            self.sess.close()
            print('Программа завершила свою работу. Спасибо за использование music genetaror')

    def _main_train(self):
        """ Цикл обучения
        """
        assert self.sess

        

        merged_summaries = tf.merge_all_summaries()
        if self.glob_step == 0:  
            self.writer.add_graph(self.sess.graph)

        print('Обучение началось (нажмите Ctrl+C для выхода и сохранения)...')

        try:  # Пытаемся сохранить на случай выхода
            e = 0
            while self.args.num_epochs == 0 or e < self.args.num_epochs:  
                e += 1

                print()
                print('------- Epoch {} (lr={}) -------'.format(
                    '{}/{}'.format(e, self.args.num_epochs) if self.args.num_epochs else '{}'.format(e),
                    self.model.learning_rp.get_lr(self.glob_step))
                )

                # Сборщик мусора
                gc.collect() 

                batches_train, batches_test = self.music_data.get_batches()

                

                tic = datetime.datetime.now()
                for next_batch in tqdm(batches_train, desc='Training'):  

                    is_output_visualized = self.glob_step % self.TRAINING_VISUALIZATION_STEP == 0

                    # передача обучения
                    ops, feed_dict = self.model.step(
                        next_batch,
                        train_set=True,
                        glob_step=self.glob_step,
                        ret_output=is_output_visualized
                    )
                    outputs_train = self.sess.run((merged_summaries,) + ops, feed_dict)
                    self.writer.add_summary(outputs_train[0], self.glob_step)

                    if is_output_visualized or (self.args.testing_curve and self.glob_step % self.args.testing_curve == 0):
                        next_batch_test = batches_test[self.glob_step % len(batches_test)]  # Генерация бачей
                        ops, feed_dict = self.model.step(
                            next_batch_test,
                            train_set=False,
                            ret_output=is_output_visualized
                        )
                        outputs_test = self.sess.run((merged_summaries,) + ops, feed_dict)
                        self.writer_test.add_summary(outputs_test[0], self.glob_step)

                    # Визуализация
                    if is_output_visualized:
                        visualization_base_name = os.path.join(self.model_dir, self.TRAINING_VISUALIZATION_DIR, str(self.glob_step))
                        tqdm.write('Visualizing: ' + visualization_base_name)
                        self._visualize_output(
                            visualization_base_name,
                            outputs_train[-1],
                            outputs_test[-1] 
                        )

                    # Промежуточное сохранение
                    self.glob_step += 1 
                    if self.glob_step % self.args.save_every == 0:
                        self._save_session(self.sess)

                toc = datetime.datetime.now()

                print('Epoch finished in {}'.format(toc-tic)) 
        except (KeyboardInterrupt, SystemExit):  # Если нажато Ctrl+C
            print('Обнаружено прерывание, выполняется выход из программы...')

        self._save_session(self.sess)  # Сохранение после обучения

    def _main_test(self):
        """ Генерация композиций
        """
        assert self.sess
        assert self.args.batch_size == 1

        print('Начинаем генерацию...')

        model_list = self._get_model_list()
        if not model_list:
            print('Внимание: не найдено модели \'{}\'. Сначала необходимо обучение'.format(self.model_dir))
            return

        batches, names = self.music_data.get_batches_test_old()
        samples = list(zip(batches, names))

        for model_name in tqdm(sorted(model_list), desc='Model', unit='model'):  
            self.saver.restore(self.sess, model_name)

            for next_sample in tqdm(samples, desc='Generating ({})'.format(os.path.basename(model_name)), unit='songs'):
                batch = next_sample[0]
                name = next_sample[1]  

                ops, feed_dict = self.model.step(batch)
                assert len(ops) == 2  
                chosen_labels, outputs = self.sess.run(ops, feed_dict)

                model_dir, model_filename = os.path.split(model_name)
                model_dir = os.path.join(model_dir, self.TESTING_VISUALIZATION_DIR)
                model_filename = model_filename[:-len(self.MODEL_EXT)] + '-' + name


                self.music_data.visit_recorder(
                    outputs,
                    model_dir,
                    model_filename,
                    [ImgConnector, midiconnect],
                    chosen_labels=chosen_labels
                )


        print('Генерация завершена, {} композиций сгенерированно.'.format(self.args.batch_size * len(model_list) * len(batches)))

    def _visualize_output(self, visualization_base_name, outputs_train, outputs_test):
        """ Запись результатов по ходу обучения.

        """

        model_dir, model_filename = os.path.split(visualization_base_name)
        for output, set_name in [(outputs_train, 'train'), (outputs_test, 'test')]:
            self.music_data.visit_recorder(
                output,
                model_dir,
                model_filename + '-' + set_name,
                [ImgConnector, midiconnect]
            )

    def _restore_previous_model(self, sess):
        """ Восстановление прошлых параметров
        """

        if self.args.test == compositor.TestMode.ALL:
            return

        print('WARNING: ', end='')

        model_name = self._get_model_name()

        if os.listdir(self.model_dir):
            if self.args.reset:
                print('Перезапуск: Удаление прошлой модели в {}'.format(self.model_dir))
            # Анализ файлов в директории
            elif os.path.exists(model_name):  # Восстановление модели
                print('Восстановление прошлой модели из {}'.format(model_name))
                self.saver.restore(sess, model_name)
                print('Модель восстановлена.')
            elif self._get_model_list():
                print('Конфликт с предыдущей моделью.')
                raise RuntimeError('Модели уже существуют в \'{}\'. Проверьте сначала их'.format(self.model_dir))
            else:  
                print('Модель не найден, но папка не пуста {}. Удаление...'.format(self.model_dir)) 
                self.args.reset = True

            if self.args.reset:
                
                for root, dirs, files in os.walk(self.model_dir, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        print('Удаление {}'.format(file_path))
                        os.remove(file_path)
        else:
            print('Предыдущая модель не найдена : {}'.format(self.model_dir))

    def _save_session(self, sess):
        """ Сохранение параметров
        """
        tqdm.write('Промежуточное сохранение: Сохраняем (не завершайте работу программы)...')
        self._save_params()
        self.saver.save(sess, self._get_model_name())  # Put a limit size (ex: 3GB for the model_dir) ?
        tqdm.write('Модель сохранена')

    def _restore_params(self):
        """ Восстановление предыдущих параметров
        """
        self.model_dir = os.path.join(self.args.root_dir, self.MODEL_DIR_BASE)
        if self.args.model_tag:
            self.model_dir += '-' + self.args.model_tag

        config_name = os.path.join(self.model_dir, self.CONFIG_FILENAME)
        if not self.args.reset and not self.args.create_dataset and os.path.exists(config_name):
            # Загрузка
            config = configparser.ConfigParser()
            config.read(config_name)

            # Проверка версии
            current_version = config['General'].get('version')
            if current_version != self.CONFIG_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(current_version, self.CONFIG_VERSION, config_name))

            # Restoring the the parameters
            self.glob_step = config['General'].getint('glob_step')
            self.args.keep_all = config['General'].getboolean('keep_all')
            self.args.dataset_tag = config['General'].get('dataset_tag')
            if not self.args.test:  # When testing, we don't use the training length
                self.args.sample_len = config['General'].getint('sample_len')

            self.args.hidden_size = config['Network'].getint('hidden_size')
            self.args.num_layers = config['Network'].getint('num_layers')
            self.args.target_weights = config['Network'].get('target_weights')
            self.args.scheduled_smp = config['Network'].get('scheduled_smp').split(' ')

            self.args.batch_size = config['Training'].getint('batch_size')
            self.args.save_every = config['Training'].getint('save_every')
            self.args.ratio_dataset = config['Training'].getfloat('ratio_dataset')
            self.args.testing_curve = config['Training'].getint('testing_curve')

            ModuleLoader.load_all(self.args, config)

            # отображение параметров
            print('Внимание: Восстановление параметров предыдущей конфигурации (Если нужны изменения - вносить вручную в настроечном файле)')

        
        if self.args.test:
            self.args.batch_size = 1
            self.args.scheduled_smp = [Model.ScheduledSamplingPolicy.NONE]

    def _save_params(self):
        """ Сохранение глобальных параметров
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version'] = self.CONFIG_VERSION
        config['General']['glob_step'] = str(self.glob_step)
        config['General']['keep_all'] = str(self.args.keep_all)
        config['General']['dataset_tag'] = self.args.dataset_tag
        config['General']['sample_len'] = str(self.args.sample_len)

        config['Network'] = {}
        config['Network']['hidden_size'] = str(self.args.hidden_size)
        config['Network']['num_layers'] = str(self.args.num_layers)
        config['Network']['target_weights'] = self.args.target_weights  # Could be modified manually
        config['Network']['scheduled_smp'] = ' '.join(self.args.scheduled_smp)

        config['Training'] = {}
        config['Training']['batch_size'] = str(self.args.batch_size)
        config['Training']['save_every'] = str(self.args.save_every)
        config['Training']['ratio_dataset'] = str(self.args.ratio_dataset)
        config['Training']['testing_curve'] = str(self.args.testing_curve)

        # Сохранение конфигурации
        ModuleLoader.save_all(config)

        with open(os.path.join(self.model_dir, self.CONFIG_FILENAME), 'w') as config_file:
            config.write(config_file)

    def _print_params(self):
        """ Печать текущих параметров
        """
        print()
        print('Текущие параметры:')
        print('glob_step: {}'.format(self.glob_step))
        print('keep_all: {}'.format(self.args.keep_all))
        print('dataset_tag: {}'.format(self.args.dataset_tag))
        print('sample_len: {}'.format(self.args.sample_len))

        print('hidden_size: {}'.format(self.args.hidden_size))
        print('num_layers: {}'.format(self.args.num_layers))
        print('target_weights: {}'.format(self.args.target_weights))
        print('scheduled_smp: {}'.format(' '.join(self.args.scheduled_smp)))

        print('batch_size: {}'.format(self.args.batch_size))
        print('save_every: {}'.format(self.args.save_every))
        print('ratio_dataset: {}'.format(self.args.ratio_dataset))
        print('testing_curve: {}'.format(self.args.testing_curve))

        ModuleLoader.print_all(self.args)
        print()

    def _get_model_name(self):
        """ Парсинг сохраняемых параметров
        Вызывается после каждого промежуточного сохранения и в самом начале
        """
        model_name = os.path.join(self.model_dir, self.MODEL_NAME_BASE)
        if self.args.keep_all:  
            model_name += '-' + str(self.glob_step)
        return model_name + self.MODEL_EXT

    def _get_model_list(self):
        """ Возвращает список моделей в директории
        """
        return [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]

    def _get_device(self):
        """ Определение железа
        """
        if self.args.device == 'cpu':
            return '"/cpu:0'
        elif self.args.device == 'gpu':  # Работает только на одной видеокарте
            return '/gpu:0'
        elif self.args.device is None:  
            return None
        else:
            print('Внимание: Ошибка в имени девайса: {}, Используйте оборудование по умолчанию'.format(self.args.device))
            return None
