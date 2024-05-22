import torch
from typing import Literal, TypedDict
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


class TensorDict(TypedDict):
  input_ids: torch.Tensor
  token_type_ids: torch.Tensor
  attention_mask: torch.Tensor


class DPR(nn.Module):
  def __init__(
      self,
      backbone_model: Literal['vinai/phobert-base-v2', 'vinai/phobert-base',
                              'vinai/phobert-large', 'FPTAI/vibert-base-cased'] = 'vinai/phobert-base-v2'
  ):
    super(DPR, self).__init__()
    self.backbone_model = backbone_model
    self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_model)
    self.question_encoder = AutoModel.from_pretrained(self.backbone_model)
    self.passage_encoder = AutoModel.from_pretrained(self.backbone_model)

  def forward(self, question_inputs: TensorDict, passage_inputs: TensorDict):
    bz = passage_inputs['input_ids'].shape[0]
    number_passage = passage_inputs['input_ids'].shape[1]

    passage_inputs['input_ids'] = passage_inputs['input_ids'].flatten(0, 1)
    passage_inputs['token_type_ids'] = passage_inputs['token_type_ids'].flatten(
        0, 1)
    passage_inputs['attention_mask'] = passage_inputs['attention_mask'].flatten(
        0, 1)

    question = self.question_encoder(**question_inputs)
    passage = self.passage_encoder(**passage_inputs)

    question = question[0]
    passage = passage[0]
    question_mask = question_inputs['attention_mask'].unsqueeze(
        -1).expand(question.size()).float()
    question = torch.sum(question * question_mask, 1)
    question = question / (question_mask.sum(1) + 1e-9)

    passage_mask = passage_inputs['attention_mask'].unsqueeze(
        -1).expand(passage.size()).float()
    passage = torch.sum(passage * passage_mask, 1)
    passage = passage / (passage_mask.sum(1) + 1e-9)
    passage = passage.view(bz, number_passage, -1)
    score = torch.bmm(passage, question)
    return score
