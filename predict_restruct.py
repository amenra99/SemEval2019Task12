from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification
import tensorflow as tf
import numpy as np
import glob
import loadData
import os

tf.random.set_seed(2019)
np.random.seed(2019)

MAX_TOKEN = 128
PRETRAINED_MODEL = 'bert-large-cased'
# val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'
val_path = "./SemEval18_Task12/Test/Test_Task2_Participants"

tf_model = TFBertForTokenClassification.from_pretrained('./save_new_saved_model/')
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

token_inputs = tf.keras.Input(shape=(None,), name='input_ids', dtype='int32')
mask_inputs = tf.keras.Input(shape=(None,), name='attention_mask', dtype='int32')
segment_inputs = tf.keras.Input(shape=(None,), name='token_type_ids', dtype='int32')

txtFiles = sorted(glob.glob(val_path + '/*.txt'))

lineTemplate = 'T{0}\tLocation {1}\t{2}\n'
annTemplate = '#{0}	AnnotatorNotes T{0}	<latlng>6.53774, 3.3522</latlng><pop>10601345</pop><geoID>2332453</geoID\n>'


def getSpans(text, tokens):
    if 'uncased' in PRETRAINED_MODEL:
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

def loadFile(txtFile, tokenizer, max_token) :
    with open(txtFile, 'rb') as file:
        text = file.read().decode("utf-8", "surrogatepass")

    tokenized_text = tokenizer.tokenize(text)
    spans = getSpans(text, tokenized_text)

    chunked_token_ids = list( [tokenizer.convert_tokens_to_ids(word) for word in divide_chunks(tokenized_text, max_token) ])
    chunked_token_ids[-1] = pad_sequence(chunked_token_ids[-1], 0, max_token)

    return chunked_token_ids, tokenized_text, spans, text

results = None
tokens = None
input_ids = None
orginSpans = None
orginText = None

# for txtFile in txtFiles[0:1]:
print(len(txtFiles))
for txtFile in txtFiles:
    print(txtFile)

    input_ids, tokens,  orginSpans, orginText = loadFile(txtFile, tokenizer, max_token=MAX_TOKEN)
    results = tf_model(np.array(input_ids, dtype=np.int32))[0]
    
    predSpans = []
    
#     for i, result in enumerate(results):
#         pred = np.argmax(result, axis=-1)
#         trues = np.where(pred == 1)
#         trues = np.concatenate(trues, axis=None)

#         for val in trues:
#             index = i * MAX_TOKEN + val
#             token = tokens[index]
#             span = orginSpans[index]
#             word = orginText[span[0]:span[1]]
#             print(token, span, word)
    
    for i, result in enumerate(results):
        pred = np.argmax(result, axis=-1)
        tmpStart = -1
        lastSpan = -1
        
        for j in range(len(pred)):
            if pred[j] > 0:
                index = i * MAX_TOKEN + j
                span = orginSpans[index]
                token = tokens[index]

                if tmpStart > 0:
                    predSpans[-1] = (lastSpan[0], span[1])
                else:
                    predSpans.append(span)
                    tmpStart = span[0]

                lastSpan = span

            else:
                tmpStart = -1
                
    with open('res/' + os.path.basename(txtFile).replace('.txt', '.ann'), 'w') as annFile:
        count = 1
        for span in predSpans:
            location = orginText[span[0]:span[1]]
            
            tmpText = location.split('\n')
            lastIndex = -1
            spanText = ''
            for word in tmpText:
                if lastIndex < 0:
                    lastIndex = span[0]
                spanText = spanText + "{0} {1};".format(lastIndex, lastIndex + len(word))
                lastIndex = lastIndex + len(word) + 1

#             print(lineTemplate.format(count, spanText[0:-1], word.replace('\n', ' ')))
            annFile.write(lineTemplate.format(count, spanText[0:-1], location.replace('\n', ' ')))
            ann.file.write(annTemplate.format(count))
            count = count + 1