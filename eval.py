import sys
sys.path.append('..')
import os
from fire import Fire
from DTgrade_model.utils.configuration import __datafile__, default_eval_model_and_config
from DTgrade_model.utils.data import DTGradeDataset
from DTgrade_model.utils.evaluation import Predictor

def get_test_predictor():
    model, config = default_eval_model_and_config()
    dataset =  DTGradeDataset.from_xml(__datafile__, model_path=config['model_path'])
    dataset.test()
    predictor = Predictor(model, dataset)
    return predictor


def eval():
    predictor = get_test_predictor()
    predictor.predictions_to_csv(os.path.join('results','test_predictions.csv'))
    predictor.metrics_to_json(os.path.join('results','test_metrics.json'))


if __name__ == '__main__':
    Fire(eval)
