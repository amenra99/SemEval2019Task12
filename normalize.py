#!/usr/bin/python

import sys
import glob
import re
import operator
import itertools

import pytorch_transformers


# path = '/Users/msk/Class/2019_Fall/directed_research/SemEval18_Task12/Training/test'
path = '/Users/msk/Class/2019_Fall/directed_research/SemEval18_Task12/Training/Training_Data_Participant'
# path = '/Users/msk/Class/2019_Fall/directed_research/SemEval18_Task12/Test/Test_Task13_Participants'

GeoIdMatch = ".*<geoID>\s*(\S+)\s*</geoID>.*"

tokenizer = pytorch_transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def main():
	# Train
	annFiles = glob.glob(path + '/*.ann')
	normalizer = readTextAndGeoIdSpans(annFiles)
	data = getInputData(normalizer)

	for obj in data:
		print(len(obj[0]))

	# Test

def getInputData(normalizer):
	for item in normalizer:
		tokens = tokenizer.tokenize(item[0])
		spans = getSpans(item[0], tokens)

		labels = [False] * len(tokens)
		for i, span in enumerate(spans):
			if span in item[1]:
				labels[i] = True

		yield tokens, labels, spans


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



main()

