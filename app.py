from flask import Flask, redirect, url_for, request, render_template, session, flash, send_file
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import subprocess
import time
import sys
import datetime
# from inltk.inltk import setup
# setup('hi')
import stanza
import string
import main
import json, os
from pydub import AudioSegment
import torch
from twilio.twiml.messaging_response import MessagingResponse
import requests
from transformers import BertTokenizer
from utils import *
# https://github.com/libindic/indic-trans
from indictrans import Transliterator
trn = Transliterator(source='eng', target='hin')


hyper_params = {
    "run_ID": 126,
    "message": "124 with different seed",
    "createData": True,
    "bs": 32,
    "BERT_FROZEN": False,
    "bert_model_name": "bert-base-multilingual-cased",
    "model_options": "ai4bharat/indic-bert, bert-base-multilingual-cased, xlm-roberta-base, bert-base-multilingual-uncased, xlm-mlm-100-1280, xlm-mlm-tlm-xnli15-1024, xlm-mlm-xnli15-1024",
    "alpha": 1e-05,
    "epochs": 5,
    "rseed": 121,
    "nw": 8,
    "cut_off": 0.51,
    "label_smoothing": False,
    "random_negative_sent": True,
    "train_ratio": 0.5,
    "val_ratio": 0.2,
    "max_len": 300,
    "data_source": "inshorts_full",
    "data_options": "asha_sentsim, inshorts_full, inshorts_400pairs",
    "architecture": "fc3",
    "arch_options": "SequenceClassification, fc1, fc2, fc3",
    "which_bert_embedding": "pooler_output",
    "bert_embedding_options": "cls, pooler_output",
    "my_datapath": "data/",
    "criterion": "BCE",
    "query_expansion": False,
    "embedding_size": 768,
	'inflate_negative_n_times': 0,
    'include_curated_negatives': False
}
query_expansion = False
which_criterion = {
    'MSE': torch.nn.MSELoss(),
    'NLL': torch.nn.NLLLoss(),
    'CrossEntropy': torch.nn.CrossEntropyLoss(),
    'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss(),
    'BCE': torch.nn.BCELoss(),
    'BCEwithLogits': torch.nn.BCEWithLogitsLoss()
}

database = {}
traindb = []

# stanza.download('hi')
try:
	nlp = stanza.Pipeline('hi', use_gpu=True)
except:
	stanza.download('hi')
	nlp = stanza.Pipeline('hi', use_gpu=True)

print("\n\n\t\t=========== Stanza model loaded ===========")



if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

bert_model_name = 'bert-base-multilingual-cased' # 'ai4bharat/indic-bert' 'bert-base-multilingual-cased' 'xlm-roberta-base' 'bert-base-multilingual-uncased' 'xlm-mlm-100-1280' (out of memory error) 'xlm-mlm-tlm-xnli15-1024' 'xlm-mlm-xnli15-1024' (poor)

mytokenizer = BertTokenizer.from_pretrained(bert_model_name,local_files_only=True)
print("\n\t\t==== Model building  =====")
mymodelST = myBert(device,hyper_params,which_criterion).to(device)
print("\n\t\t==== Model created  =====")
checkpoint = torch.load('695_epoch_3.pth.tar',map_location=torch.device(device))
print("\n\t\t==== Checkpoint loaded  =====")
print(mymodelST.load_state_dict(checkpoint['state_dict']))
del checkpoint
print("\n\n\t\t=========== MyModel loaded ===========")

# df = torch.load('allQnA.bin') # path to your bin file
if not os.path.isfile('bert_cos_model/pytorch_model.bin'):
	modelST = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
	modelST.save('bert_cos_model')
modelST = SentenceTransformer('bert_cos_model')
print("\n\n\t\t=========== SentenceTransformer loaded ===========")
# df = pd.read_csv('allQnA_small.csv')
# df['Question_a'] = df['Question'].apply(lambda d: add_word_groups(clean(d))) # when BERT_spc_added_words in on
# print("\n\n\t\t=========== Embeddings are being created ===========")
# df['embeddings'] = ''
# df['embeddings'] = list(modelST.encode(df['Question_a']))
# df['Question_clean'] = df['Question'].apply(lambda d: clean(d))
# print("\n\n\t\t=========== Embeddings created ===========")
# torch.save(df,'allQnA_small.bin')
# exit()

print("\n\n\t\t=========== Embeddings loading... =========== (takes ~20 seconds on cpu)")
df = torch.load('allQnA_small.bin')
allQnA = df
print("\n\n\t\t=========== Embeddings loaded!!! ===========")
SimModel = ''
# exit()

app = Flask(__name__)
app.secret_key = 'any random string 123q'


def cleandata(db):
    if db:
        stamp_current = int(datetime.datetime.now().timestamp())
        for x,y in db.items():
            stamp_entry = y["stamp"]
            print(stamp_current-stamp_entry)
            if stamp_current-stamp_entry > 1200:        #20 minutes
                return deletentry(x)
    return db

def getdata(content):
	url = "http://localhost:8880/bot_query"
	payload = {'question_text': content}
	response = requests.request("POST", url, data = payload)
	text = response.json()
	return text

def getdata2(content):
	# url = "http://localhost:8880/bot_query"
	# payload = {'question_text': content}
	# response = requests.request("POST", url, data = payload)

	global df
	global modelST

	df = df.dropna()

	q = content
	q  = q.translate(str.maketrans('', '', string.punctuation))

	print("\n\n\n\n\n")
	print("=========== bot_query given by user is:", q)

	# Quillpad
	# q = q.split(' ')
	# for i,x in enumerate(q):
	# 	q[i] = main.quillpad(q[i])
	# q = ' '.join(q)



	print("\n=========== bot_query transcribed by indic-trans is:", q)

	# print(q)
	# input('wait')

	df2 = df[['Question','Answer']]
	if len(q) > 0:
		# result = qc.enqueue(getTopN, df,q)
		df2 = getTopN(allQnA, q, nlp, modelST, mymodelST, SimModel, show=False)
	df2 = df2.head(3)
	print("Data Sending...")

	text = df2.to_json(orient='records')
	return text

#append to database
def appendentry(ph_key,db_value,oques):
	global database
	database[ph_key]={"data":db_value,"oques":oques,"num":0,"flag":0,"stamp":int(datetime.datetime.now().timestamp())}

def deletentry(ph_number):
	global database
	database.pop(ph_number)
	return database

#check presence
def indb(ph_number):
	global database
	try:
		database[ph_number]
	except :
		return False 
	#print("f")
	return True

def answered(ph_number):
	global database
	entry = database[ph_number]
	flag = entry["flag"]
	#qna = entry["data"]
	#q_no = entry["num"] -1 
	#oques = entry["oques"]  
	if flag == 1:
		#traindb.append([oques,qna[q_no]["Question"]])
		deletentry(ph_number)
		return True
	return False 

def appendtraindb(ph_number):
	global database
	global traindb
	entry = database[ph_number]
	qna = entry["data"]
	q_no = entry["num"] -1 
	oques = entry["oques"]
	traindb.append([oques,qna[q_no]["Question"]])

def replywithQues(ph_number):
	global database
	entry = database[ph_number]
	qna = entry["data"]
	q_no = entry["num"]
	if q_no >= len(qna):
		deletentry(ph_number)
		return "Krpiya Vistaar se prashan dobara puche"
	ques = qna[q_no]["Question"]
	entry["num"] = q_no + 1
	footer = ". Kya aap yeh puchna chate the Haan yah Na"
	return ques+footer


def replywithAns(ph_number):
	global database
	entry = database[ph_number]
	flag = entry["flag"]
	if flag == 1:
		entry["flag"] = 0
		return replywithQues(ph_number)
	appendtraindb(ph_number)
	qna = entry["data"]
	q_no = entry["num"] - 1
	ans = qna[q_no]["Answer"]
	entry["flag"] = 1
	footer = ". Kya aap aur sawaal dekna chahte hai Haan yah Na"
	return ans+footer


def fetchresult(content,ph_number):
	if not indb(ph_number):
		qna = {}
		if content.lower() != 'na' or content.lower() != 'haan': #simranjeet: this will always be true
			qna = json.loads(getdata2(content))
		else:
			msg = "Krpiya dobaara sawaal puche"
			# haan/na in 1st question
			return msg
		appendentry(ph_number,qna,content)
		ques = replywithQues(ph_number)
		return ques
	else:
		if content.lower() == 'na':
			if not answered(ph_number):
				ques = replywithQues(ph_number)
				print(ques)
				return ques
		elif content.lower() == 'haan':
			ans = replywithAns(ph_number)
			print(ans)
			return ans
		else:
			msg = "Krpiya dobaara sawaal puche"
			deletentry(ph_number)
			return msg

@app.after_request
def add_header(response):
	"""
	Add headers to both force latest IE rendering engine or Chrome Frame,
	and also to cache the rendered page for 10 minutes.
	"""
	response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
	response.headers['Cache-Control'] = 'public, max-age=0'
	return response


@app.route('/')
def hello_world():
	# df = pd.read_excel('subsetQnA.xlsx')
	df = pd.read_csv('allQnA_small.csv')
	df = df[['Question','Answer']]
	# df = df[:10]
	# file = open('q.txt','r')
	# q = file.read()
	# file.close()
	return render_template('index.html',df=df,q='')

@app.route('/enter_query', methods= ['POST'])
def enter_query():
	if request.method == 'POST':
		# df = pd.read_excel('subsetQnA.xlsx')
		# df = pd.read_csv('allQnA.csv')
		global df
		global modelST

		df = df.dropna()
		# df = df[:10]

		q = request.form['question_text']
		q  = q.translate(str.maketrans('', '', string.punctuation))


		print("\n\n\n\n\n")
		print("=========== Query given by user is:", q)

		# Quillpad
		# q = q.split(' ')
		# for i,x in enumerate(q):
		# 	q[i] = main.quillpad(q[i])
		# q = ' '.join(q)
		q = trn.transform(q)

		print("\n=========== Query transcribed by indic-trans is:", q)

		print(q)
		# q = 'E46 '+q
		# input('wait')

		df2 = df[['Question','Answer']] # this will come into play when len(q) == 0. So that system doesn't break if the user sends a blank input.
		if len(q) > 0:
			# result = qc.enqueue(getTopN, df,q)
			df2 = getTopN(allQnA, q, nlp, modelST, mymodelST, SimModel, show=False)
		return render_template('index.html',df=df2,q=q)

#http://0.0.0.0:8080/getaudio?id=47&start=1&end=15
@app.route('/getaudio', methods= ['GET'])
def fetch_audio():
	print('+'*100,'\nAudio sent request received')
	id  = int(request.args.get("id"))
	start  = float(request.args.get("start"))
	end  = float(request.args.get("end"))
	data = pd.read_csv("audio_record.csv")
	print("VAlues " +str(start)+" "+str(id)+" "+str(end))
	path = data[data.id==id].file.item() 
	audio = AudioSegment.from_wav(path) 
	start = start*1000
	end = end*1000
	chunk = audio[start:end] 
	chunk.export("clip.wav",format ="wav")
	print("Audio sending....")
	return send_file(
		 "clip.wav", 
		 mimetype="audio/wav", 
		 as_attachment=False, 
		 attachment_filename="clip.wav")



@app.route('/bot_query', methods= ['POST'])
def bot_query():
	if request.method == 'POST':
		global df
		global modelST

		df = df.dropna()

		q = request.form['question_text']
		q  = q.translate(str.maketrans('', '', string.punctuation))

		print("\n\n\n\n\n")
		print("=========== bot_query given by user is:", q)

		# Quillpad
		# q = q.split(' ')
		# for i,x in enumerate(q):
		# 	q[i] = main.quillpad(q[i])
		# q = ' '.join(q)
		q = trn.transform(q)

		print("\n=========== bot_query transcribed is:", q)

		# print(q)
		# input('wait')

		df2 = df[['Question','Answer']]
		if len(q) > 0:
			# result = qc.enqueue(getTopN, df,q)
			df2 = getTopN(allQnA, q, nlp, modelST, mymodelST, SimModel, show=False)
		df2 = df2.head(3)
		print("Data Sending...")
		return df2.to_json(orient='records')

		# =============== OLD CODE ======================
		# df =  pd.read_csv('allQnA.csv')
		# df = df.dropna()
		# # df = df[:10]
		# q = request.form['question_text']
		# q  = q.translate(str.maketrans('', '', string.punctuation))

		
		# q = q.split(' ')
		# for i,x in enumerate(q):
		# 	q[i] = main.quillpad(q[i])
		# q = ' '.join(q)

		# # print(q)
		# # input('wait')

		# if len(q) > 0:
		# 	# result = qc.enqueue(getTopN, df,q)
		# 	df = getTopN(df, q, nlp)
		# #return df['Answer'].iloc[0]
		# #print(str(df.to_json(orient='records')))
		# df = df.head(3)
		# #df.to_csv("new.csv",index=False)
		# print("Data Sending...")
		# return df.to_json(orient='records')



@app.route("/sms", methods=['POST'])
def sms_reply():
	global traindb
	global database
	database = cleandata(database.copy())
	"""Respond to incoming calls with a simple text message."""
	# Fetch the message
	print('+'*50,'\nASHA: Request received on SMS is')
	print(request)
	msg = request.form.get('Body')
	print('ASHA: message fetched', msg)
	phone_no = request.form.get('From')
	print('ASHA: phone fetched', phone_no)
	reply = fetchresult(msg,phone_no)
	print('ASHA: reply fetched')

	# Create reply
	resp = MessagingResponse()
	resp.message(reply)
	print('ASHA: Reply created')

	if traindb:
		with open('train.txt', 'a') as f:
			for item in traindb:
				f.write("%s,%s\n" %(item[0],item[1]))
		traindb = []

	return str(resp)


if __name__ == '__main__':
	app.run(host="localhost", port=8880, debug=True)
