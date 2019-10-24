#!/usr/bin/env python
# coding: utf-8
import sys
import glob
import re
import operator
import itertools
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
import loadData

train_path = './SemEval18_Task12/Training/Training_Data_Participant'
val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'

MAX_TOKEN = 512
PRETRAINED_MODEL = 'bert-base-uncased'
EPOCHS = 2
BATCH_SIZE = 2

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# obtain data train / dev
tr_inputs, tr_tags, tr_masks = loadData.getData(train_path, tokenizer)
val_inputs, val_tags, val_masks = loadData.getData(val_path, tokenizer)

# reduce data train / dev for test training
tr_inputs = tr_inputs[:10]
tr_masks = tr_masks[:10]
tr_tags = tr_tags[:10]

val_inputs = val_inputs[:3]
val_masks = val_masks[:3]
val_tags = val_tags[:3]


# create inputs
token_inputs = tf.keras.Input(shape=(None,), name='input_ids', dtype='int32')
mask_inputs = tf.keras.Input(shape=(None,), name='attention_mask', dtype='int32')
segment_inputs = tf.keras.Input(shape=(None,), name='token_type_ids', dtype='int32')

# training dataset
train_x = dict(
   input_ids = np.array(tr_inputs, dtype=np.int32),
   attention_mask = np.array(tr_masks, dtype=np.int32),
   token_type_ids = np.zeros(shape=(len(tr_inputs), MAX_TOKEN)))
train_y = np.array(tr_tags, dtype=np.int32)

# validataion dataset
val_x = dict(
   input_ids = np.array(val_inputs, dtype=np.int32),
   attention_mask = np.array(val_masks, dtype=np.int32),
   token_type_ids = np.zeros(shape=(len(val_inputs), MAX_TOKEN)))
val_y = np.array(val_tags, dtype=np.int32)


# model configuration
bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bertModel.compile(optimizer=optimizer, loss=loss, metrics=[metric])



# training
bertModel.fit(x=train_x, y=train_y, epochs=EPOCHS)

bertModel.save_pretrained('./save/')

bertModel.evaluate(x=val_x, y=val_y)



# batch training

# bertModel.fit(x=train_x, y=train_y,
#               validation_data=(val_x, val_y),
#               epochs=epochs,
#               batch_size=BATCH_SIZE)






