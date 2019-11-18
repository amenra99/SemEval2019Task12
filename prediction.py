from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification
import tensorflow as tf
import numpy as np
import glob
import loadData

tf.random.set_seed(2019)
np.random.seed(2019)

MAX_TOKEN = 256
PRETRAINED_MODEL = 'bert-base-uncased'
val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'

tf_model = TFBertForTokenClassification.from_pretrained('./save/')
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)


token_inputs = tf.keras.Input(shape=(None,), name='input_ids', dtype='int32')
mask_inputs = tf.keras.Input(shape=(None,), name='attention_mask', dtype='int32')
segment_inputs = tf.keras.Input(shape=(None,), name='token_type_ids', dtype='int32')

annFiles = glob.glob(val_path + '/*.ann')

with open('./result/result.html', 'w') as out:
    for annFile in annFiles:
        txtFile = annFile.replace('.ann', '.txt')
        lastIdx = 0
        print(txtFile)

        count = 0
        val_inputs, val_tags, val_masks, val_spans = loadData.getDataFile(annFile, tokenizer, max_token=MAX_TOKEN)
        val_x = dict(
            input_ids = np.array(val_inputs, dtype=np.int32),
            attention_mask = np.array(val_masks, dtype=np.int32),
            token_type_ids = np.zeros(shape=(len(val_inputs), MAX_TOKEN)))
        val_y = np.array(val_tags, dtype=np.int32)

        results = tf_model(val_x['input_ids'], token_type_ids=val_x['token_type_ids'])[0]

        with open(txtFile, 'r') as tFile:
            txt = tFile.read().decode("utf-8", "surrogatepass")
            out.write('<h3>')
            out.write(txtFile)
            out.write('</h3><p>')
            for i, result in enumerate(results):
                pred = np.argmax(result, axis=-1)
                for j, token in enumerate(tokenizer.convert_ids_to_tokens(val_inputs[i])):

                    if val_tags[i][j] == 1 and pred[j] == 1:
                        span = val_spans[count]
                        out.write(txt[lastIdx:span[0]])
                        out.write("<b><font color='green'>")
                        out.write(txt[span[0]:span[1]])
                        out.write('</font></b>')
                        lastIdx = span[1]
                    elif val_tags[i][j] == 1:
                        span = val_spans[count]
                        out.write(txt[lastIdx:span[0]])
                        out.write("<b><font color='blue'>")
                        out.write(txt[span[0]:span[1]])
                        out.write('</font></b>')
                        lastIdx = span[1]
                    elif pred[j] == 1:
                        span = val_spans[count]
                        out.write(txt[lastIdx:span[0]])
                        out.write("<b><font color='red'>")
                        out.write(txt[span[0]:span[1]])
                        out.write('</font></b>')
                        lastIdx = span[1]
                        
                    count += 1
                    
            out.write(txt[lastIdx:]) #flush
            out.write('</p><br><br><br>') #end of document


