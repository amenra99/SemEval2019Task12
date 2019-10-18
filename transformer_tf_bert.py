#!/usr/bin/env python
# coding: utf-8
import sys
import glob
import re
import operator
import itertools
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification
from sklearn.model_selection import train_test_split

MAX_TOKEN = 512
epochs = 2
batch_size = 2

train_path = './SemEval18_Task12/Training/Training_Data_Participant'
val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'
GeoIdMatch = ".*<geoID>\s*(\S+)\s*</geoID>.*"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def readTextAndGeoIdSpans(annFiles):

    for annFile in annFiles:
        txtFile = annFile.replace('.ann', '.txt')

        with open(txtFile, 'rb') as file:
                text = file.read().decode("utf-8", "surrogatepass")

        annIdToSpan = {}
        spanToGeoID = {}

        with open(annFile, 'r') as file:
            annText = file.read()
            annText = re.sub(r"\n([^T#])", r" ", annText)
            lines = re.split('\r|\n', annText)
            pos = 0

            for line in lines:
                line = line.strip()
                row = line.split('\t')
                if len(row) > 0:
                    if row[0].startswith('T'):
                        if row[1].startswith('Location'):
                            position = row[1].replace(';', ' ').split()
                            startP = int(position[1])
                            endP = int(position[-1])

                            locationText = text[startP:endP]

                            if re.sub(r'\s+', '', locationText) != re.sub(r'\s+', '', row[2]):
                                print("WARNING: {0}({1}) from .txt != {2}({3}:{4}) from {5}".format(
                                    list(locationText), str(text).find(row[2], pos+1), row[2], startP, endP, annFile[-14:]))

                            pos = endP
                            span = (startP, endP)
                            annIdToSpan[row[0]] = span

                        elif row[1].startswith('Protein'):
                            pass

                    elif row[0].startswith('#'):
                        noteId = row[1].split()[1]

                        # if noteId in annIdToSpan:
                        match = re.search(GeoIdMatch, row[2])
                        if match:
                            geoId = match.groups()[0].strip()
                            spanToGeoID[noteId] = geoId
                        # else:
                        # 	print(noteId)


        spans = sorted(annIdToSpan.items(), key=operator.itemgetter(1))
        geoIds = sorted(spanToGeoID.items(), key=lambda i:[j[0] for j in spans].index(i[0]))

        # try:
        # 	geoIds = sorted(spanToGeoID.items(), key=lambda i:[j[0] for j in spans].index(i[0]))
        # # print(geoIds)
        # except:
        # 	print("error in {0}".format(annFile))
            # exit(0)

        yield text, [i[1] for i in spans], [i[1] for i in geoIds]
        # exit(0)
        

def getInputData(tokenizer, normalizer):
    docs = []
    for item in normalizer:
        tokens = tokenizer.tokenize(item[0])
        spans = getSpans(item[0], tokens)

#         labels = [False] * len(tokens)
        labels = [0] * len(tokens)
        for i, span in enumerate(spans):
            if span in item[1]:
#                 labels[i] = True
                labels[i] = 1                

    # 		yield tokens, labels, spans
        docs.append([tokens, labels, spans])
    return docs


def getSpans(text, tokens):
    text = text.lower()
    spans = []
    end = 0

    for token in tokens:
        token = token.replace('#', '')
        start = text.find(token, end)
        end = start + len(token)
        spans.append((start, end))

    return spans


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]
        
def pad_sequence(l, v, n):
    tmp = [v] * n;
    for i in range(0, len(l)):
        tmp[i] = l[i]
    return tmp

    
def chunk_data(data):
    tokens = []
    labels = []

    for doc in data:
        tokens = list( [tokenizer.convert_tokens_to_ids(txt) for txt in divide_chunks(doc[0], MAX_TOKEN) ])
        labels = list(divide_chunks(doc[1], MAX_TOKEN))

        tokens[-1] = pad_sequence(tokens[-1], 0, MAX_TOKEN)
        labels[-1] = pad_sequence(labels[-1], 0, MAX_TOKEN)
        
        tokens.extend(tokens)
        labels.extend(labels)        
#         yield tokens, labels
    return tokens, labels
        
def getData(path):
    annFiles = glob.glob(path + '/*.ann')
    normalizer = readTextAndGeoIdSpans(annFiles)
    data = getInputData(tokenizer, normalizer)  

    tokens, labels = chunk_data(data)
    attention_masks = [[float(i>0) for i in ii] for ii in tokens]

    return tokens, labels, attention_masks

# obtain data train / dev
tr_inputs, tr_tags, tr_masks = getData(train_path)
val_inputs, val_tags, val_masks = getData(val_path)


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
bertModel = TFBertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bertModel.compile(optimizer=optimizer, loss=loss, metrics=[metric])



# training
bertModel.fit(x=train_x, y=train_y, epochs=epochs)

bertModel.save_pretrained('./save/')

bertModel.evaluate(x=val_x, y=val_y)



# batch training

# bertModel.fit(x=train_x, y=train_y,
#               validation_data=(val_x, val_y),
#               epochs=epochs,
#               batch_size=batch_size)






