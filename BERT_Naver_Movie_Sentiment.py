import tensorflow as tf
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

train = pd.read_csv("ratings_train.txt", sep="\t") ## Train_FileName
test = pd.read_csv("ratings_test.txt", sep="\t")   ## Test_FileName

#문장만 출력
sentences = train['document']
sentences = ["[CLS]" + str(sentence) + "[SEP]" for sentence in sentences]
##라벨만 출력
labels = train['label'].values

###bert-base-multilingual-cased을 이용하여 워드피스 ex) 나는 밥을 먹었다 --> 나 #는 밥 #을 먹 #었다.
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = False)
tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]

####문장 길이 맞추기 위해 padding값 추가하기
max_sequence_length = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=max_sequence_length, dtype="long", truncating="post", padding="post")

##어텐션마스크 초기화, 패딩이면 0, 패딩이 아니면 1
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0)  for i in seq]
    attention_masks.append(seq_mask)

#학습데이터(문장,라벨)를 훈련셋과 검증셋으로 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels,
                                                                                    random_state=2018,
                                                                                    test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=2018,
                                                       test_size=0.1)


# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

#파이토치의 텐서로 변환된 데이터 sample 출력
print(train_inputs[0])
print(train_labels[0])
print(train_masks[0])
print(validation_inputs[0])
print(validation_labels[0])
print(validation_masks[0])

##데이터셋 만들기
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler = validation_sampler, batch_size = batch_size)



#####test data 전처리 (Train Data와 동일하게 전처리 함)
print(test[0:5])
print(test.shape)

sentences = test['document']
sentences = ["[CLS]" + str(sentence) + "[SEP]" for sentence in sentences]
print(sentences[0:5])

labels = test['label'].values
print(labels[0:5])


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case = False)
tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]

print(sentences[0])
print(tokenized_texts[0])

max_sequence_length = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=max_sequence_length, dtype="long", truncating="post", padding="post")
print(input_ids[0])

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0)  for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)
