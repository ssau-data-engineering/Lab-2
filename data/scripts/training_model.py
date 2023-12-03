import pandas as pd

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Callable
from tqdm.notebook import tqdm
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from transformers import pipeline, AutoTokenizer, AutoModel, BertTokenizer, BertModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from collections import Counter, defaultdict

import re
from pymorphy2 import MorphAnalyzer

# чтение датафрейма (ограничим датасет тк на CPU обучение полного датасета будет слишком долго идти)
df = pd.read_csv('/data/output/ml_data.csv', sep='\t')
df.head()

# названия категорий которые будем оставлять в датафрейме
CATEGORY_NAMES = df.category.unique()

# служебные словари
INDEX_TO_CLASS = dict(enumerate(CATEGORY_NAMES))
CLASS_TO_INDEX = {value: key for key, value in INDEX_TO_CLASS.items()}

model_path = 'cointegrated/rubert-tiny'
tokenizer = AutoTokenizer.from_pretrained(model_path)  # do_lower_case=True

# токенизация и сопоставление индексов токенов с добавлением служебных токенов
encoded = tokenizer.encode('Всем привет!', add_special_tokens=True)

max_calc_len = 0
for sent in df.text:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_calc_len = max(max_calc_len, len(input_ids))

print(f'Максимальная длина токенизированного текста: {max_calc_len}')

# Кодирование текста

# макс длина текста в токенах
max_len = 71

# списки для индексов токенов, масок внимания и тагрет индексов категорий текста
input_ids = []
attention_masks = []
target_ids = []

# итерация по строкам всех столбцов датафрейма
for row in df.itertuples():
    # текст из столбца text
    text = row.text
    encoded_dict = tokenizer.encode_plus(
                        text,  # текст строка которую кодируем
                        add_special_tokens=True,  # добавить '[CLS]' и '[SEP]' токены
                        max_length=max_len,  # параметр максимальной длины текста
                        padding='max_length',  # делать падинг до макс длины
                        truncation=True,  # если длина больше max_length то отрезать лишнее
                        return_attention_mask=True,  # делать ли маску внимания
                        return_tensors='pt',  # формат возвращаемого тензора
                        # return_token_type_ids=False,
                   )

    # обновить списки
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    target_id = CLASS_TO_INDEX[row.category]
    target_ids.append(target_id)

# преобразовать списки в тензоры
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
target_ids = torch.tensor(target_ids)

print('Оригинальный текст: ', text)
print('Токенизированный текст:', encoded_dict['input_ids'])
print('Класс текста:', INDEX_TO_CLASS[target_id])


# датасет из тензоров
dataset = TensorDataset(input_ids, attention_masks, target_ids)

# разделение на датасеты для обучения и валидации
val_size = 0.2
train_dataset, val_dataset = random_split(dataset, [1 - val_size, val_size])

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # pin_memory=True
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # pin_memory=True

input_ids, attention_mask, target_ids = next(iter(train_loader))

# инициализация модели с необходимым кол-вом классов
num_labels = len(CLASS_TO_INDEX)
model_name = 'cointegrated/rubert-tiny'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# получение батча из даталоадера
input_ids, attention_mask, target_ids = next(iter(train_loader))

# берт подобные модели на вход обычно принимают индексы токенов и маски внимания
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# расчет ошибки
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(outputs.logits, target_ids)

print(loss)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)

# Train
##### get_linear_schedule_with_warmup для изменения скорости обучения оптимизатора
from transformers import get_linear_schedule_with_warmup

torch.manual_seed(111)
# кол-во классов
num_labels = len(CLASS_TO_INDEX)
# название модели
model_path = 'cointegrated/rubert-tiny'
# инициализация модели
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

# оптимизатор с маленькой скоростью потому что трансформер и потому что шедулер будет ее увеличивать
LR = 0.000_02
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# кол-во эпох обучения
EPOCHS = 40
# расчет общего кол-ва шагов обучения для шедулера
total_steps = len(train_loader) * EPOCHS

# инициализация планировщика - шедулера
scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * total_steps)

# функция ошибки
loss_fn = torch.nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)


# Обучение всей модели с использованием шедулера и градиент клипинга
# обучалось на T4 в Colab
torch.manual_seed(111)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(EPOCHS):
    # ========================= TRAIN ===============================
    model.train()

    correct_predictions = 0
    epoch_loss = 0

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    for input_ids, attention_mask, targets in train_loader:
        # переместить все тензоры на девайс
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        targets = targets.to(DEVICE)

        # передать в модель индексы токенов и макси внимания
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # получение ответов модели для расчета accuracy и расчет ошибки
        preds = torch.argmax(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, targets)

        epoch_loss += loss.item()
        correct_predictions += torch.sum(preds == targets).item()

        # обучение моедли
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # нормировка ошибки accuracy
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct_predictions / len(train_loader.dataset)

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # ==================== VALIDATION ============================
    model.eval()

    correct_predictions = 0
    epoch_loss = 0

    for input_ids, attention_mask, targets in val_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = torch.argmax(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, targets)

        epoch_loss += loss.item()
        correct_predictions += torch.sum(preds == targets).item()

    val_loss = epoch_loss / len(train_loader)
    val_acc = correct_predictions / len(val_loader.dataset)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    print("=" * 50)

## Inference
@torch.inference_mode()
def predict_text(model, text):
    encoded_dict = tokenizer.encode_plus(
                        text,  # текст строка которую кодируем
                        add_special_tokens=True,  # добавить '[CLS]' и '[SEP]' токены
                        truncation=True,  # если длина больше 512 то отрезать лишнее
                        return_attention_mask=True,  # делать ли маску внимания
                        return_tensors='pt',  # формат возвращаемого тензора
                        # return_token_type_ids=False,
                   )
    # переместить тензоры на девайс
    encoded_dict = {key: value.to(DEVICE) for key, value in encoded_dict.items()}
    outputs = model(**encoded_dict)
    pred_tokens_ids = outputs.logits.argmax(-1).item()
    print(f'Класс текста: {INDEX_TO_CLASS[pred_tokens_ids]}')

# перевести модель в режим оценки
model.eval();

text = 'Выгодно купить'
predict_text(model, text)

# Сохраняем модель
torch.save(model, 'model_air_test.pth')