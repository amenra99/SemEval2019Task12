-------------
Introduction
-------------
The accompanying data comes from PubMed Central (PMC) Open Access articles and contains two files. the files contain data generated from distant supervision for identifying geographic locations, as described in the articles:
- https://www.ncbi.nlm.nih.gov/pubmed/28815119
- https://www.ncbi.nlm.nih.gov/pubmed/29950020

-------------
Format
-------------
Both files contain IO encoded tab delimited contents where the first column is the word/token and the second column is the token encoding (I-LOC). The files haven't been tokenized by sentences. 

-------------
Permissions
-------------
You are free to use this data and choose to modify/enhance the encodings in any shape or form to help train your models on more data.

-------------
Disclaimer
-------------
NOTE: These are weakly supervised encodings and using these as-is without filtering may reduce the performance of the full system as they are not gold-standard annotations. Hence, use the data with caution.

-------------
File List
-------------
1) pos.txt - This file contains sentences that contain positive examples i.e. geographic locations. Due to the nature of distant supervision used, not all geographic locations have been detected and encoded. However, all locations encoded are likely to be geographic locations with a few exceptions of course, due to the nature of data generation.

2) neg.txt - This file contains sentences that most likely do not contain geographic locations. Due to the noisy nature of the data, it may contain a few false negative examples.

