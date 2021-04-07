import pandas as pd
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    DataLoader)
import xml.etree.ElementTree as ElementTree
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import NamedTuple
from types import SimpleNamespace
from transformers import AutoTokenizer
from tqdm import tqdm
try:
    from utils.configuration import __datafile__, __default_model_path__
except ModuleNotFoundError:
    from configuration import __datafile__, __default_model_path__
import torch
import re

class DTGradeInstance(NamedTuple):
    ID: int
    Label: int
    LabelString: str
    ProblemDescription: str
    Question: str
    Answer: str
    ReferenceAnswers: list
    MetaInfo: dict

    @staticmethod
    def from_xml(instance):
        ID = instance.attrib['ID']
        for child in instance:
            if child.tag == 'Annotation':
                LabelString = child.attrib['Label']
                try:
                    Label = int(DTGradeDataset.label_to_class[LabelString])
                except KeyError:
                    Label = -99
            if child.tag == 'ProblemDescription':
                ProblemDescription = child.text
            if child.tag == 'Question':
                Question = child.text
            if child.tag == 'Answer':
                Answer = child.text
            if child.tag == 'ReferenceAnswers':
                ReferenceAnswers = child.text.split('\n')
                ReferenceAnswers = [re.sub('^[0-9]*:[  \t]*', '', r) for r in ReferenceAnswers if r != '']
                ReferenceAnswers = list(ReferenceAnswers)
            if child.tag == 'MetaInfo':
                MetaInfo = child.attrib
        return DTGradeInstance(int(ID), Label, LabelString,  ProblemDescription, Question, Answer, ReferenceAnswers, MetaInfo)

    @staticmethod
    def from_df(df):
        ID = df['ID'].unique().item()
        Label = df['Label'].unique().item()
        ProblemDescription = df['ProblemDescription'].unique().item()
        Question = df['Question'].unique().item()
        Answer = df['Answer'].unique().item()
        ReferenceAnswers = df['ReferenceAnswer']
        MetaInfo = {}
        LabelString = ''
        return DTGradeInstance(int(ID), Label, LabelString,  ProblemDescription, Question, Answer, ReferenceAnswers, MetaInfo)


    def explode(self):
        return [{'ID': self.ID,
                 'Label': self.Label,
                 'ProblemDescription': self.ProblemDescription,
                 'Question': self.Question,
                 'Answer': self.Answer,
                 'ReferenceAnswer': ref_answer} for ref_answer in self.ReferenceAnswers]

    def to_df(self):
        return pd.DataFrame.from_records(self.explode())

class DTGradeDataset(Dataset):
    label_to_class = {
        # This is kinda ugly, but most explicit I could come up with
        'correct(1)|correct_but_incomplete(0)|contradictory(0)|incorrect(0)': 0,
        'correct(0)|correct_but_incomplete(1)|contradictory(0)|incorrect(0)': 1,
        'correct(0)|correct_but_incomplete(0)|contradictory(1)|incorrect(0)': 2,
        'correct(0)|correct_but_incomplete(0)|contradictory(0)|incorrect(1)': 3
        }

    def __init__(self, instances, drop_dirty = True, train_test_IDs = None,  model_path = __default_model_path__, percent = 100):
        self.num_labels = 4
        self.instances = instances
        self.data = self.get_df(drop_dirty=drop_dirty)
        self.percent = percent
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, lowercase = True)
        self.encode()
        self._data = self.data.copy()
        self.instanceIDs = np.array([instance.ID for instance in instances])
        if train_test_IDs is None:
            self._trainIDs, self._testIDs = train_test_split(self._data['ID'].unique(), test_size=0.2, random_state=42)
        else:
            self._trainIDs, self._testIDs = train_test_IDs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def encode(self):
        EncodedText = [None] * len(self.data)
        for i in tqdm(range(len(self.data)), desc="Encoding text"):
            row = self.data.iloc[i]
            problem_tokens = self.tokenizer.encode(row['ProblemDescription'])
            question_tokens = self.tokenizer.encode(row['Question'])
            reference_tokens =  self.tokenizer.encode(row['ReferenceAnswer'])
            answer_tokens = self.tokenizer.encode(row['Answer'])
            tokens = problem_tokens + question_tokens[1:] + reference_tokens[1:] + answer_tokens[1:]
            EncodedText[i] = torch.Tensor(tokens).long()
        self.data['input_ids'] = EncodedText

    def take_percentage(self):
        if self.percent < 100:
            num_rows = int(self.percent*0.01*len(self.data))
            self.data = self.data.iloc[:num_rows]

    def train(self):
        self.data = self._data[self._data['ID'].isin(self._trainIDs)]
        self.take_percentage()

    def test(self):
        self.data = self._data[self._data['ID'].isin(self._testIDs)]
        self.take_percentage()

    def reset(self):
        self.data = self._data

    def get_df(self, drop_dirty = True, ID = None):
        records = []
        for instance in self.instances:
            records += instance.explode()
        df = pd.DataFrame.from_records(records)
        if drop_dirty:
            df = df[df['Label'] != -99]
        if ID is not None:
            df = df[df['ID']==ID]
        return df

    def get_instance_by_ID(self, ID):
        idx = self.instanceIDs.tolist().index(ID)
        return self.instances[idx]

    def get_train_instances(self):
        return [instance for instance, ID in zip(self.instances, self.instanceIDs) if ID in self._trainIDs]

    def get_test_instances(self):
        return [instance for instance, ID in zip(self.instances, self.instanceIDs) if ID in self._testIDs]


    @staticmethod
    def from_xml(path, **kwargs):
        tree = ElementTree.parse(path)
        root = tree.getroot()
        instances = [DTGradeInstance.from_xml(instance) for instance in root]
        return DTGradeDataset(instances, **kwargs)

    @staticmethod
    def collater(batch):
        input_ids = [b.input_ids for b in batch]
        labels = [b.Label for b in batch]
        data = {'input_ids':pad_tensor_batch(input_ids),
                'labels': torch.Tensor(labels).long()}
        return Batch(**data)


class Batch(SimpleNamespace):
    def __init__(self, **kwargs):
        super(Batch, self).__init__(**kwargs)

    def cuda(self):
        atts = self.__dict__
        for att, val in atts.items():
            try:
                self.__dict__[att] = val.cuda()
            except AttributeError:
                pass

    def cpu(self):
        atts = self.__dict__
        for att, val in atts.items():
            try:
                self.__dict__[att] = val.cpu()
            except AttributeError:
                pass

    def __contains__(self, item):
        return item in self.__dict__

    def generate_mask(self):
        assert "input_ids" in self
        return torch.where(self.input_ids.eq(0), self.input_ids, torch.ones_like(self.input_ids))



def pad_tensor_batch(tensors, pad_token = 0):
    max_length = max([t.size(0) for t in tensors])
    batch = torch.zeros((len(tensors), max_length)).long()
    if pad_token > 0:
        batch.fill_(pad_token)
    for i, tensor in enumerate(tensors):
        batch[i, :tensor.size(0)] = tensor
    return batch


def get_train_dataloader(datafile = __datafile__,
                         model_path = __default_model_path__,
                         num_workers = 4 if torch.cuda.is_available() else 0,
                         percent = 100,
                         batch_size = 1,
                         drop_last = False):
    dataset = DTGradeDataset.from_xml(datafile,
                                      model_path=model_path,
                                      percent=percent
                                      )
    dataset.train()
    print(f"Training set loaded with {len(dataset.data)} lines of data.")
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size = batch_size, drop_last=drop_last)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset.collater, num_workers = num_workers)
    return loader


def get_test_dataloader(datafile = __datafile__,
                         model_path = __default_model_path__,
                         num_workers = 4 if torch.cuda.is_available() else 0,
                         percent = 100,
                         batch_size = 1,
                         drop_last = False):
    dataset = DTGradeDataset.from_xml(datafile,
                                      model_path=model_path,
                                      percent=percent
                                      )
    dataset.test()
    print(f"Test set loaded with {len(dataset.data)} lines of data.")
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size = batch_size, drop_last=drop_last)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dataset.collater, num_workers = num_workers)
    return loader
