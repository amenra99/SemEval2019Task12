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
EPOCHS = 20
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# obtain data train / dev
tr_inputs, tr_tags, tr_masks = loadData.getData(train_path, tokenizer, MAX_TOKEN)
val_inputs, val_tags, val_masks = loadData.getData(val_path, tokenizer, MAX_TOKEN)

## reduce data train / dev for test training
# tr_inputs = tr_inputs[:10]
# tr_masks = tr_masks[:10]
# tr_tags = tr_tags[:10]

# val_inputs = val_inputs[:3]
# val_masks = val_masks[:3]
# val_tags = val_tags[:3]


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


# Callback to calculate precision / recall / F1 score
class EvaluateModel(tf.keras.callbacks.Callback):
    def __init__(self, train_x, tr_tags, val_x, val_tags):
        self.train_x = train_x
        self.tr_tags = tr_tags
        self.val_x = val_x
        self.val_tags = val_tags

        self.reports = []
        # print('init')

    def on_epoch_end(self, epoch, logs={}):

        tr_result, tr_eval = self.getEvalSummary(self.train_x, self.tr_tags)
        val_result, val_eval = self.getEvalSummary(self.val_x, self.val_tags)
        self.reports.append([[tr_result, tr_eval], [val_result, val_eval]])
        # trainPred, trainRecall, trainF1 = self.getEvalSummary(self.train_x, self.tr_tags)
        # testPred, testRecall, recallF1 = self.getEvalSummary(self.val_x, self.val_tags)
        # self.reports.append([[trainPred, trainRecall, trainF1], [testPred, testRecall, recallF1]])

        print('\nEpoch {0}\tTr_Precision: {1}\t Tr_Recall: {2}\t Tr_F1: {3}\tVal_Precision: {4}\t Val_Recall: {5}\t Val_F1: {6}'.format(
            epoch, tr_eval[0], tr_eval[1], tr_eval[2], val_eval[0], val_eval[1], val_eval[2]))
        print(self.reports)


    def getEvalSummary(self, x, y):
      corrects, preds, trues = 0, 0, 0
      results = self.model(x['input_ids'], token_type_ids=x['token_type_ids'])[0]

      for i, result in enumerate(results):
          pred = np.argmax(result, axis=-1)

          tmp_correct, tmp_preds, tmp_trues = self.getCorrects(result, np.array(y[i]))
          corrects += tmp_correct
          preds += tmp_preds
          trues += tmp_trues

          print(corrects, preds, trues)

      precision = corrects/preds if corrects > 0 else 0
      recall = corrects/trues  if corrects > 0 else 0
      f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

      return [corrects, preds, trues], [precision, recall, f1]


    def getCorrects(self, preds, labels):
        pred_flat = np.argmax(preds, axis=-1).flatten()
        labels_flat = labels.flatten()  # labels as integers

        con1 = (pred_flat == 1)
        con2 = (labels_flat == 1)

        part = np.where(con1 & con2)
        correct = len(part[0])
        sum_pred = np.sum(pred_flat)
        sum_true = np.sum(labels_flat)

        return correct, sum_pred, sum_true

    def get(self):
        return self.reports


# model configuration
bertModel = TFBertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # labels as integer
# loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # only two labels (one-hot)
# loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # two or more one-hot encoding
# metric = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
eval_metrics = EvaluateModel(train_x, tr_tags, val_x, val_tags)
# bertModel.compile(optimizer=optimizer, loss=loss, metrics=metric)
bertModel.compile(optimizer=optimizer, loss=loss)



# training
bertModel.fit(x=train_x, y=train_y, epochs=EPOCHS, callbacks=[eval_metrics])

# batch training
# bertModel.fit(x=train_x, y=train_y,
#               validation_data=(val_x, val_y),
#               epochs=EPOCHS,
#               # steps_per_epoch=115,
#               # validation_steps=7,
#               batch_size=BATCH_SIZE)


reports = eval_metrics.get()
# print(reports)

print('[tr_result, tr_eval], [val_result, val_eval]')
for report in reports:
  print(report)

with open('bert_result.csv', 'w') as f:
  f.write('tr_corr, tr_pred, tr_true, tr_precision, tr_recall, tr_f1, val_corr, val_pred, val_true, val_precision, val_recall, val_f1\n')
  for tr_val in reports:  # [0]training   [1]validation
    for result_eval in tr_val:  # [0]result   [1]eval data
      for item in result_eval: # [0]corrects, preds, trues   [1]precision, recall, f1
        for val in item:
          f.write(str(val))
          f.write(',')
    f.write('\n')
    

# save bert model
bertModel.save_pretrained('./save/')


# evaluation
# bertModel.evaluate(x=val_x, y=val_y)





