from utils.configuration import __datafile__, default_eval_model_and_config
from utils.data import DTGradeDataset
from utils.evaluation import Predictor

def eval():
    model, config = default_eval_model_and_config()
    dataset =  DTGradeDataset.from_xml(__datafile__, model_path=config['model_path'])
    dataset.test()
    predictor = Predictor(model, dataset)
    df = predictor.predict_data()
    j =  predictor.compute_metrics()
    print(df)
    print(j)
