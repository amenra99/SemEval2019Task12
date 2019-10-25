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

tf.random.set_seed(2019)
np.random.seed(2019)

train_path = './SemEval18_Task12/Training/Training_Data_Participant'
val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'

MAX_TOKEN = 256
PRETRAINED_MODEL = 'bert-base-uncased'
EPOCHS = 30
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# obtain data train / dev
tr_inputs, tr_tags, tr_masks = loadData.getData(train_path, tokenizer, MAX_TOKEN)
val_inputs, val_tags, val_masks = loadData.getData(val_path, tokenizer, MAX_TOKEN)

# reduce data train / dev for test training
# tr_inputs = tr_inputs[:30]
# tr_masks = tr_masks[:30]
# tr_tags = tr_tags[:30]

# val_inputs = val_inputs[:10]
# val_masks = val_masks[:10]
# val_tags = val_tags[:10]


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


# import transformers

# class BERT(transformers.TFBertModel):
#    def __init__(self, config, *inputs, **kwargs):
#        super(BERT, self).__init__(config, *inputs, **kwargs)
#        self.bert.call = tf.function(self.bert.call)

# # bert = BERT.from_pretrained(PRETRAINED_MODEL)
# bert = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=1)
# token_encodings = bert([token_inputs, mask_inputs, segment_inputs])[0]
# # Keep [CLS] token encoding
# sentence_encoding = tf.squeeze(token_encodings[:, 0:1, :], axis=1)
# # Apply dropout
# sentence_encoding = tf.keras.layers.Dropout(0.1)(sentence_encoding)
# # Final output (projection) layer
# outputs = tf.keras.layers.Dense(MAX_TOKEN, activation='sigmoid', name='outputs')(sentence_encoding)
# # Wrap-up model
# model = tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])
# # Compile model
# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# loss = tf.keras.losses.BinaryCrossentropy()
# # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# model.compile(optimizer=optimizer, loss=loss)

# model.fit(x=train_x, y=train_y, epochs=EPOCHS)

# model.save_weights("./tmp_save/model.h5")


# model configuration
bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # labels as integer
# loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # only two labels (one-hot)
# loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # two or more one-hot encoding
# metric = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
# bertModel.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bertModel.compile(optimizer=optimizer, loss=loss)
# bertModel.compile(optimizer=optimizer, loss=loss, metrics=metric)


# training
# bertModel.fit(x=train_x, y=train_y, epochs=EPOCHS)

# batch training
bertModel.fit(x=train_x, y=train_y,
              validation_data=(val_x, val_y),
              epochs=EPOCHS,
              # steps_per_epoch=115,
              # validation_steps=7,
              batch_size=BATCH_SIZE)


bertModel.save_pretrained('./save/')

# bertModel.evaluate(x=val_x, y=val_y)




