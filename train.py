from scheduler import LinearWarmupScheduler
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

train = DPRDataset("./data/bkgpt.train.json", use_word_segmentation=True)
train = DataLoader(dataset=train, batch_size=16, shuffle=True)
test = DPRDataset("./data/bkgpt.test.json")
test = DataLoader(dataset=test, batch_size=16, use_word_segmentation=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
total_steps = epochs * len(train)
warmup_steps = int(total_steps * 0.1)
scheduler = LinearWarmupScheduler(optimizer, warmup_steps, total_steps)

gradient_steps = 1
for epoch in range(epochs):
  train_step = 0
  train_loss = 0
  train_loop = tqdm(train)
  number_train_batch = len(train_loop)
  model.train()
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
    loss = loss / gradient_steps
    train_loss += loss.item() * gradient_steps
    loss.backward()
    if (batch_idx + 1) % gradient_steps == 0 or (batch_idx + 1) == number_train_batch:
      optimizer.step()
      optimizer.zero_grad()
      scheduler.step()
    train_loop.set_description(
        f"Epoch [{epoch+1}/{epochs}] Training loss: {log(train_loss / train_step)} Lr: {log(optimizer.param_groups[0]['lr'], 6)}"
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
