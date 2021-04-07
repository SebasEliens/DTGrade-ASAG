from types import SimpleNamespace
import os
import yaml
import torch
import transformers
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


def load_model_from_disk(path):
    weights, config = torch.load(path, map_location='cpu')
    config = config['_items']
    mdl = transformers.AutoModelForSequenceClassification.from_pretrained(config['model_path'], num_labels = config['num_labels'])
    mdl.load_state_dict(weights)
    return mdl, config


def default_eval_model_and_config():
    model_dir =  os.path.join(os.path.dirname(__file__),'..', 'models', __default__['eval']['modeldir'])
    for f in os.listdir(model_dir):
        if f.endswith('.pt'):
            model, config = load_model_from_disk(os.path.join(model_dir, f))
            break
    return model, config
