import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, TypedDict
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import underthesea


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

  def next_batch(self):
    pass
