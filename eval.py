from utils.configuration import __datafile__, default_eval_model_and_config
from utils.data import DTGradeDataset
from utils.evaluation import Predictor
import os

def eval():
    model, config = default_eval_model_and_config()
    dataset =  DTGradeDataset.from_xml(__datafile__, model_path=config['model_path'])
    dataset.test()
    predictor = Predictor(model, dataset)
    predictor.predictions_to_csv(os.path.join('results/test_predictions.csv'))
    predictor.metrics_to_json(os.path.join('results/test_metrics.json'))
