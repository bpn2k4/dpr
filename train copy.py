from model import DPR
from data import DPRDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch


def log(x, length=5):
  return "{:.{}f}".format(x, length)


device = 'cuda:0'
model = DPR()
model.to(device)

epochs = 5

train = DPRDataset("./data/bkgpt.train.json")
train = DataLoader(dataset=train, batch_size=16, shuffle=True)
test = DPRDataset("./data/bkgpt.test.json")
test = DataLoader(dataset=test, batch_size=16)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

accumulation_steps = 8
for epoch in range(epochs):
  train_step = 0
  train_loss = 0
  train_loop = tqdm(train)
  model.train()
  optimizer.zero_grad()
  for batch_idx, batch in enumerate(train_loop):
    train_step += 1
    question_inputs = {
        "input_ids": batch['question_input_ids'].to(device),
        "token_type_ids": batch['question_token_type_ids'].to(device),
        "attention_mask": batch['question_attention_mask'].to(device),
    }
    passage_inputs = {
        "input_ids": batch['passage_input_ids'].to(device),
        "token_type_ids": batch['passage_token_type_ids'].to(device),
        "attention_mask": batch['passage_attention_mask'].to(device),
    }
    output = model(question_inputs, passage_inputs)
    label = batch['label'].to(device)
    loss = loss_fn(output, label)
    loss = loss / accumulation_steps
    train_loss += loss.item()
    loss.backward()
    if (batch_idx + 1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
    train_loop.set_description(
        f"Epoch [{epoch+1}/{epochs}] Training loss: {log(train_loss / train_step)}"
    )

  test_loss = 0
  test_step = 0
  test_loop = tqdm(test)
  model.eval()
  with torch.no_grad():
    for batch in test_loop:
      test_step += 1
      question_inputs = {
          "input_ids": batch['question_input_ids'].to(device),
          "token_type_ids": batch['question_token_type_ids'].to(device),
          "attention_mask": batch['question_attention_mask'].to(device),
      }
      passage_inputs = {
          "input_ids": batch['passage_input_ids'].to(device),
          "token_type_ids": batch['passage_token_type_ids'].to(device),
          "attention_mask": batch['passage_attention_mask'].to(device),
      }
      output = model(question_inputs, passage_inputs)
      label = batch['label'].to(device)
      loss = loss_fn(output, label)
      test_loss += loss.item()
      test_loop.set_description(
          f"Epoch [{epoch+1}/{epochs}] Testing loss: {log(test_loss / test_step)}"
      )
