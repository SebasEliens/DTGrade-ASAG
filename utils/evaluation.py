import pandas as pd
import json
from torch.nn.functional import softmax
from tqdm import tqdm
try:
    from utils.data import DTGradeDataset
    from utils.training import metrics
except ModuleNotFoundError:
    from data import DTGradeDataset
    from training import metrics

class Predictor:
    def __init__(self, model, dataset):
        self.dataset = dataset
        self.model = model
        self.predictions = None
        self.metrics = None

    def predict_data(self):
        predictions = []
        for ID in tqdm(self.dataset.data['ID'].unique(), desc = 'Compute predictions'):
            df = self.dataset.data[self.dataset.data['ID'] == ID]
            Label = df['Label'].unique().item()
            batch = DTGradeDataset.collater([df.iloc[i] for i in range(len(df))])
            logits = self.model(input_ids = batch.input_ids, attention_mask = batch.generate_mask()).logits
            pred = softmax(logits, dim = -1).mean(0).argmax(-1).item()
            predictions.append({'ID': ID, 'Label': Label, 'Pred': pred})
        df = pd.DataFrame.from_records(predictions)
        self.predictions = df
        return df

    def predictions_to_csv(self, path):
        if not isinstance(self.predictions, pd.DataFrame):
            self.predict_data()
        self.predictions.to_csv(path)

    def compute_metrics(self):
        if not isinstance(self.predictions, pd.DataFrame):
            self.predict_data()
        y_true = self.predictions['Label']
        pred = self.predictions['Pred']
        metric_params_weighted = {'average':'weighted', 'labels':list(range(self.dataset.num_labels))}
        metric_params_macro =  {'average':'macro', 'labels':list(range(self.dataset.num_labels))}
        p,r,f1,acc = metrics(pred, y_true, metric_params_weighted)
        p_m,r_m,f1_m,acc_m = metrics(pred, y_true, metric_params_macro)
        mtr =  {'weighted-F1': f1, 'macro-F1': f1_m, 'weighted-accuracy': acc, 'macro-accuracy': acc_m }
        self.metrics = mtr
        return mtr

    def metrics_to_json(self, path):
        if not self.metrics:
            self.compute_metrics()
        with open(path, 'w') as f:
            json.dump(self.metrics, f)
