""" 
Запись всех модулей
Код частично адаптирован из https://github.com/Conchylicultor/MusicGenerator
"""

from musicsource.modulemanager import ModuleManager

from musicsource.modules import batchbuilder
from musicsource.modules import learningratepolicy
from musicsource.modules import encoder
from musicsource.modules import decoder
from musicsource.modules import loopprocessing


class ModuleLoader:
    """ Модуль-менеджер
    """
    enco_cells = None
    deco_cells = None
    batch_builders = None
    learning_rate_policies = None
    loop_procs = None

    @staticmethod
    def register_all():
        """ Список доступных модулей
        """
        ModuleLoader.batch_builders = ModuleManager('batch_builder')
        ModuleLoader.batch_builders.register(batchbuilder.Relative)
        ModuleLoader.batch_builders.register(batchbuilder.PianoRoll)

        ModuleLoader.learning_rate_policies = ModuleManager('learning_rate')
        ModuleLoader.learning_rate_policies.register(learningratepolicy.Cst)
        ModuleLoader.learning_rate_policies.register(learningratepolicy.StepsWithDecay)
        ModuleLoader.learning_rate_policies.register(learningratepolicy.Adaptive)

        ModuleLoader.enco_cells = ModuleManager('enco_cell')
        ModuleLoader.enco_cells.register(encoder.Identity)
        ModuleLoader.enco_cells.register(encoder.Rnn)
        ModuleLoader.enco_cells.register(encoder.Embedding)

        ModuleLoader.deco_cells = ModuleManager('deco_cell')
        ModuleLoader.deco_cells.register(decoder.Lstm)
        ModuleLoader.deco_cells.register(decoder.Perceptron)
        ModuleLoader.deco_cells.register(decoder.Rnn)

        ModuleLoader.loop_procs = ModuleManager('loop_proc')
        ModuleLoader.loop_procs.register(loopprocessing.SampleSoftmax)
        ModuleLoader.loop_procs.register(loopprocessing.ActivateScale)

    @staticmethod
    def save_all(config):
        """ Сохранение конфигурации модулей
        """
        config['Modules'] = {}
        ModuleLoader.batch_builders.save(config['Modules'])
        ModuleLoader.learning_rate_policies.save(config['Modules'])
        ModuleLoader.enco_cells.save(config['Modules'])
        ModuleLoader.deco_cells.save(config['Modules'])
        ModuleLoader.loop_procs.save(config['Modules'])

    @staticmethod
    def load_all(args, config):
        """ Восстановление конфигурации модулей
        """
        ModuleLoader.batch_builders.load(args, config['Modules'])
        ModuleLoader.learning_rate_policies.load(args, config['Modules'])
        ModuleLoader.enco_cells.load(args, config['Modules'])
        ModuleLoader.deco_cells.load(args, config['Modules'])
        ModuleLoader.loop_procs.load(args, config['Modules'])

    @staticmethod
    def print_all(args):
        """ Вывод модулей в текущей конфигурации
        """
        ModuleLoader.batch_builders.print(args)
        ModuleLoader.learning_rate_policies.print(args)
        ModuleLoader.enco_cells.print(args)
        ModuleLoader.deco_cells.print(args)
        ModuleLoader.loop_procs.print(args)
