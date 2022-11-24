from sentence_transformers import SentenceTransformer, util
import pandas as pd
import subprocess
import time
import sys
import re
import datetime
from tqdm import tqdm
# from inltk.inltk import setup
# setup('hi')
import stanza
import string
import main
import json
from pydub import AudioSegment
import torch
from twilio.twiml.messaging_response import MessagingResponse
import requests
from transformers import BertTokenizer
from utils import getTopN, myBert
from tqdm import tqdm
from indictrans import Transliterator
trn = Transliterator(source='eng', target='hin')


nlp = stanza.Pipeline('hi', use_gpu=False)
print("\n\n\t\t=========== Stanza model loaded ===========")
modelST = SentenceTransformer('distilbert-base-nli-mean-tokens').to('cpu')
print("\n\n\t\t=========== SentenceTransformer loaded ===========")

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

mytokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
print("\n\t\t==== Model building  =====")
mymodelST = myBert(device).to(device)
print("\n\t\t==== Model created  =====")
checkpoint = torch.load('19.pth.tar',map_location=torch.device(device))
print("\n\t\t==== Checkpoint loaded  =====")
print(mymodelST.load_state_dict(checkpoint['state_dict']))
print("\n\n\t\t=========== MyModel loaded ===========")

df1 = torch.load('allQnA.bin') # path to your bin file
print("\n\n\t\t=========== Embedding file loaded ===========")

replacedict = {
	'बेनिफिट्स':'बेनिफिट्स फायदे',
	'विटामिन्स': 'विटामिन्स पोषण',
	'इंजेक्शन': 'इंजेक्शन टीके',
	'वाइट':'वाइट सफ़ेद',
	'ब्लड प्रेशर': 'ब्लड प्रेशर बीपी',
	'स्वेलिंग': 'स्वेलिंग सूजन',
	'शुगर':'शुगर डायबिटीज',
	'डेंजर':'डेंजर खतरा',
	'आयल':'आयल तेल',
	'जौंडिस': 'जौंडिस पीलिया',
	'वक्सीनशन': 'वक्सीनशन टीकाकरण',
	'स्टार्ट': 'स्टार्ट शुरू',
	'सोप': 'सोप साबुन',
	'हनी': 'हनी शहद',
	'मीनोपॉज':'मीनोपॉज रजोनिवृत्ति',
	'अटैचमेंट': 'अटैचमेंट लगाव',
	'फीवर': 'फीवर बुखार',
	'पेरासिटामोल': 'पेरासिटामोल पीसीएम',
	'टेम्परेचर': 'टेम्परेचर तापमान',
	'पीरियड': 'पीरियड मासिक माहवारी', #newly added
	"दूध": 'दूध फीड ब्रेस्टफीड स्तनपान',
	"शुगर": ' '.join(["शुगर","डायबिटीज","मधुमय"]),
	"छाती": ' '.join(["छाती","स्तन","स्तनों","ब्रैस्ट"])
}


colname = 'Paraphrased' #'Question'
fname = 'allQuestions2' #'topic_wise'

df = pd.read_csv(fname+'.csv')
df = df[df[colname].notnull()].reset_index(drop=True)
df = df.drop(columns=['Question'])


df['m1 Suggested Q1'], df['m1 Q1 answer'], df['m1 Q1 score'], df['m1 Suggested Q2'], df['m1 Q2 answer'], df['m1 Q2 score'], df['m1 Suggested Q3'], df['m1 Q3 answer'], df['m1 Q3 score'] = '', '', '', '', '', '', '', '', ''
df['m2 Suggested Q1'], df['m2 Q1 answer'], df['m2 Q1 score'], df['m2 Suggested Q2'], df['m2 Q2 answer'], df['m2 Q2 score'], df['m2 Suggested Q3'], df['m2 Q3 answer'], df['m2 Q3 score'] = '', '', '', '', '', '', '', '', ''
df['m3 Suggested Q1'], df['m3 Q1 answer'], df['m3 Q1 score'], df['m3 Suggested Q2'], df['m3 Q2 answer'], df['m3 Q2 score'], df['m3 Suggested Q3'], df['m3 Q3 answer'], df['m3 Q3 score'] = '', '', '', '', '', '', '', '', ''

mnames = ['गंगा', 'काली', 'रेनू']

for i in tqdm(range(len(df))):
	q = df.iloc[i][colname]
	q  = q.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
	q = re.sub(r'\s+',' ',q)
	trans_string = q.maketrans("१२३४५६७८९०","1234567890")
	q = q.translate(trans_string)
	q = trn.transform(q)

	for k,v in zip(replacedict.keys(), replacedict.values()):
		q = q.replace(k,v)

	df.at[i,colname] = q

	for k, mn in enumerate(mnames):
		q2 = mn+' '+q
		df2 = getTopN(df1, q2, nlp, modelST, mymodelST, mytokenizer, show=False).iloc[:3].reset_index(drop=True)

		for j in range(len(df2)):
			df.at[i, 'm'+str(k+1)+' Suggested Q'+str(j+1)] = df2.iloc[j]['Question']
			df.at[i,'m'+str(k+1)+' Q'+str(j+1)+' answer'] = df2.iloc[j]['Answer']
			df.at[i, 'm'+str(k+1)+' Q'+str(j+1)+' score'] = df2.iloc[j]['score']

print(df)
df.to_csv(fname+'_results.csv', index=None)



