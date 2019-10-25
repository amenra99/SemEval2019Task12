from transformers import BertTokenizer, TFBertForTokenClassification, BertForTokenClassification
import tensorflow as tf
import numpy as np
import loadData

tf.random.set_seed(2019)
np.random.seed(2019)

MAX_TOKEN = 256

val_path = './SemEval18_Task12/Training/Validation_Data_Codalab/detection'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# pytorch_model = BertForTokenClassification.from_pretrained('./save/', from_tf=True)
tf_model = TFBertForTokenClassification.from_pretrained('./save/')

# get dev data
val_inputs, val_tags, val_masks = loadData.getData(val_path, tokenizer, MAX_TOKEN)



# val_inputs = val_inputs[:30]
# val_masks = val_masks[:30]
# val_tags = val_tags[:30]


# create inputs
token_inputs = tf.keras.Input(shape=(None,), name='input_ids', dtype='int32')
mask_inputs = tf.keras.Input(shape=(None,), name='attention_mask', dtype='int32')
segment_inputs = tf.keras.Input(shape=(None,), name='token_type_ids', dtype='int32')

# validataion dataset
val_x = dict(
	input_ids = np.array(val_inputs, dtype=np.int32),
	attention_mask = np.array(val_masks, dtype=np.int32),
	token_type_ids = np.zeros(shape=(len(val_inputs), MAX_TOKEN)))
val_y = np.array(val_tags, dtype=np.int32)

# predict
results = tf_model(val_x['input_ids'], token_type_ids=val_x['token_type_ids'])[0]

# print(results)
# print(np.argmax(results, axis=-1).flatten())
# exit(0)


def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=-1).flatten()
	labels_flat = labels.flatten()	# labels as integers
	# labels_flat = np.argmax(labels, axis=-1).flatten()	# one-hot encoding
	
	correct = np.sum(pred_flat == labels_flat)

	accuracy = correct / len(labels_flat)
	precision = correct / np.sum(pred_flat)
	recall = correct / np.sum(labels_flat)

	# print(pred_flat, '\n', labels_flat)
	# print(correct, accuracy, precision, recall)
	# exit(0)

	return accuracy, precision, recall


nb_eval_steps, eval_accuracy, eval_precision, eval_recall = 0, 0, 0, 0
predictions , true_labels = [], []

for i, result in enumerate(results):
	# pred = [list(p) for p in np.argmax(result, axis=-1)]
	pred = np.argmax(result, axis=-1)
	# print(np.array(pred).flatten())

	# predictions.extend(pred)
	true_labels.append(np.array(val_tags[i]))

	tmp_eval_accuracy, tmp_eval_precision, tmp_eval_recall = flat_accuracy(result, np.array(val_tags[i]))
	eval_accuracy += tmp_eval_accuracy
	eval_precision += tmp_eval_precision
	eval_recall += tmp_eval_recall
	nb_eval_steps += 1

print("Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Precision: {}".format(eval_precision/nb_eval_steps))
print("Recall: {}".format(eval_recall/nb_eval_steps))
