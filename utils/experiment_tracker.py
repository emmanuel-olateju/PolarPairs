# @title Experiment Class

from typing import List

import os
import yaml


class Experiment:

    def __init__(self, name, version, dir, description, messages=[], baseline=None, author=None, date=None):
        self.name = name
        self.version = version
        self.dir = dir
        self.description = description
        self.messages = messages
        self.parameters = dict()
        self.baseline = baseline
        self.author = author
        self.date = date

        os.makedirs(self.dir, exist_ok=True)

        if os.path.exists(f'{self.dir}/{self.version}_analysis_log.txt'):
            os.remove(f'{self.dir}/{self.version}_analysis_log.txt')

    def update_param(self, parameter: 'Parameter'):
        var_name = parameter.get_var_name()
        parameter_class = parameter.get_parameter_class()
        value = parameter.get_value()

        # assert isinstance(
        #     value,
        #     (
        #         int, float, str, dict, list,
        #         np.ndarray, torch.tensor, type(None)
        #     )
        # )

        if (parameter_class is None) or (parameter_class.lower() == 'global'):
            self.parameters[var_name] = value
            return

        if parameter_class not in self.parameters:
            self.parameters[parameter_class] = dict()
        self.parameters[parameter_class][var_name] = value

    def save(self):
        print(self.parameters)
        experiment_dict = {
            'name': self.name,
            'version': self.version,
            'baseline': self.baseline,
            'description': self.description,
            'author': self.author,
            'date': self.date,
            'messages': self.messages,
            'parameters': self.parameters,
        }

        with open(f'{self.dir}/{self.version}.yaml', "w") as f:
            yaml.dump(experiment_dict, f, default_flow_style=False)

        print(f"Model saved to {self.dir}")

    def add_params(self, parameters: List['Parameter']):

        for parameter in parameters:
            self.update_param(parameter)

    def dict_2_params(self, params_dict, params_class):
        params_list = []
        for key, value in params_dict.items():
            params_list.append(Parameter(value, key, params_class))

        self.add_params(params_list)


class Parameter:

    def __init__(self, value, var_name, parameter_class):
        self.__var_name = var_name
        self.__value = value
        self.__parameter_class = parameter_class

    def get_var_name(self):
        return self.__var_name

    def get_parameter_class(self):
        return self.__parameter_class

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value

    def set_var_name(self, var_name):
        self.__var_name = var_name

    def set_parameter_class(self, parameter_class):
        self.__parameter_class = parameter_class