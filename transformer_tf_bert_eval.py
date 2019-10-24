from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification
import tensorflow as tf
import numpy as np
import loadData

val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# pytorch_model = BertForTokenClassification.from_pretrained('./save/', from_tf=True)
tf_model = TFBertForTokenClassification.from_pretrained('./save/')


# get dev data
val_inputs, val_tags, val_masks = loadData.getData(val_path, tokenizer)

for i in range(0, len(val_inputs)):
    tensor = tf.constant(val_inputs[i])[None, :]
    result = tf_model(test)
    print([list(p) for p in np.argmax(result, axis=-1)])