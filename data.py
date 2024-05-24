import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, TypedDict
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import underthesea
import random


class DPRDataset(Dataset):
  def __init__(
      self,
      path="",
      question_length=64,
      passage_length=256,
      k=5,
      use_word_segmentation=False,
      backbone_model: Literal['vinai/phobert-base-v2', 'vinai/phobert-base',
                              'vinai/phobert-large', 'FPTAI/vibert-base-cased'] = 'vinai/phobert-base-v2',
  ):
    super(DPRDataset, self).__init__()
    self.question_length = question_length
    self.passage_length = passage_length
    self.backbone_model = backbone_model
    self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_model)
    self.path = path
    self.k = k
    self.use_word_segmentation = use_word_segmentation
    self.load_data()

  def load_data(self):
    with open(self.path, "r", encoding="utf-8") as f:
      data = json.loads(f.read())
    self.question_dict = {}
    self.chunk_dict = {}
    self.data = []
    for i in tqdm(data):
      question_id = i['question_id']
      chunk_id = i['chunk_id']
      chunk = i['chunk']
      question = i['question']
      answer = i['answer']
      file = i['file']
      if chunk_id not in self.chunk_dict:
        if self.use_word_segmentation:
          text = underthesea.word_tokenize(chunk, format="text")
        else:
          text = chunk
        output = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.passage_length
        )
        self.chunk_dict[chunk_id] = {
            "chunk_id": chunk_id,
            "text": chunk,
            "input_ids": output['input_ids'][0].tolist(),
            "token_type_ids": output['token_type_ids'][0].tolist(),
            "attention_mask": output['attention_mask'][0].tolist(),
            "file": file
        }
      if question_id not in self.question_dict:
        if self.use_word_segmentation:
          text = underthesea.word_tokenize(question, format="text")
        else:
          text = question
        output = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=self.question_length
        )
        self.question_dict[question_id] = {
            "question_id": question_id,
            "chunk_id": chunk_id,
            "text": question,
            "input_ids": output['input_ids'][0].tolist(),
            "token_type_ids": output['token_type_ids'][0].tolist(),
            "attention_mask": output['attention_mask'][0].tolist(),
            "file": file
        }
    for i in tqdm(data):
      question_id = i['question_id']
      chunk_id_ = i['chunk_id']
      file = i['file']
      negative = [
          chunk_id for chunk_id in self.chunk_dict
          if self.chunk_dict[chunk_id]['file'] != file
      ]
      self.data.append({
          "question_id": question_id,
          "positive": chunk_id_,
          "negative": negative
      })

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data[index]
    question = self.question_dict[item['question_id']]
    positive = self.chunk_dict[item['positive']]
    negative = random.sample(item['negative'], self.k)
    negative = [self.chunk_dict[i] for i in negative]
    negative = [positive] + negative
    label = [0 for _ in range(self.k + 1)]
    label[0] = 1
    data = {
        "question_input_ids": torch.tensor(question['input_ids']),
        "question_token_type_ids": torch.tensor(question['token_type_ids']),
        "question_attention_mask": torch.tensor(question['attention_mask']),
        "passage_input_ids": torch.stack([torch.tensor(i['input_ids']) for i in negative]),
        "passage_token_type_ids": torch.stack([torch.tensor(i['token_type_ids']) for i in negative]),
        "passage_attention_mask": torch.stack([torch.tensor(i['attention_mask']) for i in negative]),
        "label": torch.tensor(label, dtype=torch.float32)
    }
    return data

  def sample(self, index):
    return self.__getitem__(index)


if __name__ == "__main__":
  dataset = DPRDataset('./data/bkgpt.test.json')
  sample = dataset.sample(0)
  print(sample['question_input_ids'].shape)
  print(sample['question_token_type_ids'].shape)
  print(sample['question_attention_mask'].shape)
  print(sample['passage_input_ids'].shape)
  print(sample['passage_token_type_ids'].shape)
  print(sample['passage_attention_mask'].shape)
