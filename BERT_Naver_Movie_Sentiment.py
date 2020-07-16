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
