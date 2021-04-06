from types import SimpleNamespace
import os
import yaml

__default_config_file__ = os.path.join(os.path.dirname(__file__), '..', 'configs', 'defaults.yml')

def load_configs_from_file(file_path):
    configs = dict()
    with open(file_path, 'r') as f:
        y = yaml.load_all(f, Loader = yaml.FullLoader)
        for d in y:
            configs.update(d)
    return configs

__default__ = load_configs_from_file(__default_config_file__)
__datafile__ =  os.path.join(os.path.dirname(__file__),'..',  __default__['data']['directory'],  __default__['data']['filename'])
__default_model_path__ = __default__['training']['model_path']

def train_config():
    return SimpleNamespace(**(__default__['training']))
