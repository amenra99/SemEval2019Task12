
# coding: utf-8

import sys
import glob
import re
import operator
import itertools

from tqdm import tqdm, trange

import pytorch_transformers
from pytorch_transformers import BertForTokenClassification

from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

MAX_TOKEN = 512
epochs = 2
max_grad_norm = 1.0
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


path = './SemEval18_Task12/Training/Training_Data_Participant'
GeoIdMatch = ".*<geoID>\s*(\S+)\s*</geoID>.*"



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
        


def getInputData(normalizer):
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


tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-uncased')

annFiles = glob.glob(path + '/*.ann')
normalizer = readTextAndGeoIdSpans(annFiles)
data = getInputData(normalizer)    



def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]
        
def pad_sequence(l, v, n):
    tmp = [v] * n;
    for i in range(0, len(l)):
        tmp[i] = l[i]
    return tmp

input_tokens = []
input_labels = []
    
def chunk_data(data):
    for doc in data:
        tokens = list( [tokenizer.convert_tokens_to_ids(txt) for txt in divide_chunks(doc[0], MAX_TOKEN) ])
        labels = list(divide_chunks(doc[1], MAX_TOKEN))

        tokens[-1] = pad_sequence(tokens[-1], 0, MAX_TOKEN)
        labels[-1] = pad_sequence(labels[-1], 0, MAX_TOKEN)
        
        input_tokens.extend(tokens)
        input_labels.extend(labels)        
#         yield tokens, labels
        

chunk_data(data)
attention_masks = [[float(i>0) for i in ii] for ii in input_tokens]


# split data train / dev
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_tokens, input_labels, 
                                                            random_state=2019, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_tokens,
                                             random_state=2019, test_size=0.1)


# build tensors
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)



train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)




# model initialize
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)




FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


# In[190]:


from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



# Train model
for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss, _ = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# Evaluation

model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[27]:


# import torch

# from pytorch_transformers.modeling_bert import BertConfig, BertForPreTraining, load_tf_weights_in_bert


# tf_checkpoint_path="pretrained_bert/model.ckpt"
# bert_config_file = "bert-base-cased-config.json"
# pytorch_dump_path="pytorch_bert"

# config = BertConfig.from_json_file(bert_config_file)
# print("Building PyTorch model from configuration: {}".format(str(config)))
# model = BertForPreTraining(config)

# # Load weights from tf checkpoint
# load_tf_weights_in_bert(model, config, tf_checkpoint_path)

# # Save pytorch-model
# print("Save PyTorch model to {}".format(pytorch_dump_path))
# torch.save(model.state_dict(), pytorch_dump_path)


# In[28]:




