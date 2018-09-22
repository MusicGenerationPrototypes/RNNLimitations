""" 
Модуль-менеджер, управление классами
Код частично адаптирован из https://github.com/Conchylicultor/MusicGenerator
"""

from collections import OrderedDict


class ModuleManager:
    """ 
    """
    def __init__(self, name):
        """
        Args:
            name (str): the name of the module manager (useful for saving/printing)
        """

        self.modules = OrderedDict()  # The order on which the modules are added is conserved
        self.module_inst = None  # Reference to the chosen module
        self.module_name = ''  # Type of module chosen (useful when saving/loading)
        self.module_param = []  # Arguments passed (for saving/loading)
        self.name = name

    def register(self, module):
        """ Регистрация нового модуля
        Args:
            module (Class): the module class to register
        """
        assert not module.get_module_id() in self.modules  # Overwriting not allowed
        self.modules[module.get_module_id()] = module

    def get_modules_ids(self):
        """ Список добавленных модулей
        Returns:
            list[str]: the list of modules
        """
        return self.modules.keys()

    def get_chosen_name(self):
        """ Возвращает имя выбранного модуля
        Returns:
            str: the name of the chosen module
        """
        return self.module_name

    def get_module(self):
        """ Возвращает выбранный модуль
        Returns:
            Obj: the reference on the module instance
        """
        assert self.module_inst is not None
        return self.module_inst

    def build_module(self, args):
        """ Инициализация выбранного модуля
        Args:
            args (Obj): the global program parameters
        Returns:
            Obj: the created module
        """
        assert self.module_inst is None

        module_args = getattr(args, self.name)  

        self.module_name = module_args[0]
        self.module_param = module_args[1:]
        self.module_inst = self.modules[self.module_name](args, *self.module_param)
        return self.module_inst

    def add_argparse(self, group_args, comment):
        """ Добавление модуля в парсер командной строки
        Args:
            group_args (ArgumentParser):
            comment (str): help to add
        """
        assert len(self.modules.keys())   

        keys = list(self.modules.keys())
        group_args.add_argument(
            '--{}'.format(self.name),
            type=str,
            nargs='+',
            default=[keys[0]],  
            help=comment + ' Choices available: {}'.format(', '.join(keys))
        )

    def save(self, config_gr):
        """ Сохранение конфигурации модулей
        Args:
            config_gr (dict): dictionary where to write the configuration
        """
        config_gr[self.name] = ' '.join([self.module_name] + self.module_param)

    def load(self, args, config_gr):
        """ Восстановление параметров модулей
        Args:
            args (parse_args() returned Obj): the parameters of the models (will be modified)
            config_gr (dict): the module group parameters to extract
        Warning: Only restore the arguments. The instantiation is not done here
        """
        setattr(args, self.name, config_gr.get(self.name).split(' '))

    def print(self, args):
        """ Вывод текущей конфигурации модулей
        Args:
            args (parse_args() returned Obj): the parameters of the models
        """
        print('{}: {}'.format(
            self.name,
            ' '.join(getattr(args, self.name))
        ))
