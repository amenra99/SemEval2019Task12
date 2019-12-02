#!/usr/bin/env python
# coding: utf-8
import sys
import glob
import re
import operator
import itertools

GeoIdMatch = ".*<geoID>\s*(\S+)\s*</geoID>.*"

# word span calculate
def getSpanWord(span, text):
    locs = text.split()
    start = int(span[0])
    end = int(span[1])
    spans = []
    if len(locs) > 1:
        for loc in locs:
            result = getSpanWord([start, start + len(loc)], loc)
            start = start + len(loc) + 1
            spans.append(result)
    else:
        spans.append(((start, end), text))
        # print(text)
    return spans


def readTextAndGeoIdSpansFile(annFile):
    txtFile = annFile.replace('.ann', '.txt')

    with open(txtFile, 'rb') as file:
            text = file.read().decode("utf-8", "surrogatepass")

    annIdToSpan = {}
    spanToGeoID = {}


    # read file
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

                        row[1] = row[1][9:] # after location
                        positions = []
                        if ';' in row[1]:
                            spans = row[1].split(';')
                            lastPos = 0
                            for span in spans:
                                span = span.split()
                                result = getSpanWord(span, row[2][lastPos:lastPos + int(span[1]) - int(span[0])])
                                lastPos = int(span[1]) - int(span[0]) + 1
                                positions.extend(result)
                        else:
                            spans = row[1].split()
                            result = getSpanWord(spans, row[2])
                            positions.extend(result)
                            
                        for i, pos in enumerate(positions):
                            if type(pos) is list:
                                pos = pos[0]
                            annIdToSpan["{0}-{1}".format(row[0], i)] = pos[0]

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
                    #   print(noteId)


    spans = sorted(annIdToSpan.items(), key=operator.itemgetter(1))
    # geoIds = sorted(spanToGeoID.items(), key=lambda i:[j[0] for j in spans].index(i[0]))

    return text, [i[1] for i in spans] #, [i[1] for i in geoIds]


def readTextAndGeoIdSpans(annFiles):
    docs = []
    for annFile in annFiles:
        docs.append(readTextAndGeoIdSpansFile(annFile))
    return docs


def getInputData(tokenizer, normalizer):
    docs = []
    for item in normalizer:
        tokens = tokenizer.tokenize(item[0])
        spans = getSpans(item[0], tokens)

        labels = [0] * len(tokens)
        # labels = []  # one-hot encoding
        for i, span in enumerate(spans):
            if span in item[1]:
                labels[i] = 1
                # labels.append([0, 1])  # one-hot encoding
            # else:
                # labels.append([1, 0])  # one-hot encoding

    #       yield tokens, labels, spans
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

    
def chunk_data(data, tokenizer, max_token):
    tot_tokens = []
    tot_labels = []
    tot_spans = []

    for doc in data:
        tokens = list( [tokenizer.convert_tokens_to_ids(txt) for txt in divide_chunks(doc[0], max_token) ])
        labels = list(divide_chunks(doc[1], max_token))

        tokens[-1] = pad_sequence(tokens[-1], 0, max_token)
        labels[-1] = pad_sequence(labels[-1], 0, max_token)
        # labels[-1] = pad_sequence(labels[-1], [1, 0], max_token) # one-hot encoding
        
        
        tot_tokens.extend(tokens)
        tot_labels.extend(labels)  
        tot_spans.extend(doc[2])      
#         yield tokens, labels
    return tot_tokens, tot_labels, tot_spans

# From Path
def getData(path, tokenizer, max_token=512):
    annFiles = glob.glob(path + '/*.ann')
    normalizer = readTextAndGeoIdSpans(annFiles)
    data = getInputData(tokenizer, normalizer)  

    tokens, labels, spans = chunk_data(data, tokenizer, max_token)
    attention_masks = [[float(i>0) for i in ii] for ii in tokens]

    return tokens, labels, attention_masks, spans

# From File
def getDataFile(file, tokenizer, max_token=512):
    normalizer = [readTextAndGeoIdSpansFile(file)]
    data = getInputData(tokenizer, normalizer)  

    tokens, labels, spans = chunk_data(data, tokenizer, max_token)
    attention_masks = [[float(i>0) for i in ii] for ii in tokens]

    return tokens, labels, attention_masks, spans



# from transformers import BertTokenizer
# import numpy as np
# MAX_TOKEN = 256

# # val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'
# val_path = './SemEval18_Task12/temp'
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# val_inputs, val_tags, val_masks, val_spans = getData(val_path, tokenizer, max_token=MAX_TOKEN)

# # print(val_tags[0:5])
# # print(val_spans)

# for i, val in enumerate(val_tags[1]):
#     if val == 1:
#         print(val_spans[i + MAX_TOKEN])

