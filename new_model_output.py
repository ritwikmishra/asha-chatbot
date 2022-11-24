'''
this code generates output for any of the 4 models
i) BERT_spc
ii) BERT_cos
iii) DTP
iv) BERT_sim
'''
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
from transformers import BertTokenizer, XLMRobertaTokenizer, AutoTokenizer, AutoModel
from utils import getTopN, myBert, add_word_groups, SiameseNetwork
from tqdm import tqdm
from indictrans import Transliterator
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# export CUDA_DEVICE_ORDER=PCI_BUS_ID # this works
# os.environ['CUDA_VISIBLE_DEVICES']="2"
trn = Transliterator(source='eng', target='hin')
# stanza.download('hi')


import time
import torch
import psutil
def stathere():
    av = []
    if torch.cuda.is_available():
        av.append(torch.cuda.memory_allocated(torch.device("cuda"))/(1024*1024*1024))
    else:
        av.append(0)
    av.append(psutil.virtual_memory().available/(1024*1024*1024))
    a = time.time()
    return av, a

def statnow(av, a):
    info = ''
    if torch.cuda.is_available():
        print("Memory taken on GPU", round(torch.cuda.memory_allocated(torch.device("cuda"))/(1024*1024*1024)-av[0],3), "GB")
        info = "Memory taken on GPU "+str(round(torch.cuda.memory_allocated(torch.device("cuda"))/(1024*1024*1024)-av[0],3))+" GB\n"
    print("Memory taken on RAM", round(av[1]-(psutil.virtual_memory().available/(1024*1024*1024)),3),"GB")
    info += "Memory taken on RAM "+str(round(av[1]-(psutil.virtual_memory().available/(1024*1024*1024)),3))+" GB\n"
    print(round(time.time()-a), "seconds taken")
    info += str(round(time.time()-a))+" seconds taken"
    return info



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

import os, torch, random, numpy as np
os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
# Torch RNG
torch.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed_all(hyper_params['rseed'])
# Python RNG
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])

def clean(hi_str):
	puncts = string.punctuation+"‟“’❝❞‚‘‛❛❜❟"
	trans_string = hi_str.maketrans("१२३४५६७८९०","1234567890")
	hi_str = hi_str.translate(trans_string)
	hi_str = re.sub(r'[a-z]+','',hi_str.lower())
	hi_str = hi_str.translate(str.maketrans(puncts,' '*len(puncts))).lower().replace('|','').replace('।','')
	hi_str = re.sub(r'\s+',' ',hi_str).strip()
	return hi_str

mn = 'BERT_spc'
suff = ''
checkpointname = '695_epoch_3'

import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda:0")
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t -(c-a)  # free inside cache
    print("GPU is available", torch.cuda.get_device_name(), round(t/(1024*1024*1024)), "GB")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

av, a = stathere()

nlp = stanza.Pipeline('hi', use_gpu=True) if mn in ['DTP','Lem','E'] else ''
print("\n\n\t\t=========== Stanza model loaded ===========")
modelST = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2').to(device) if mn in ['BERT_cos','E'] else ''
print("\n\n\t\t=========== SentenceTransformer loaded ===========")


def key_words(s):
	doc = nlp(s)
	ml = []
	for sent in doc.sentences:
		for word in sent.words:
			if word.pos in ['NOUN','PROPN','VERB']:
				ml.append(word.text)
	return ' '.join(ml)

print("\n\t\t==== Model building  =====")
mymodelST = myBert(device,hyper_params,which_criterion).to(device) if mn in ['BERT_spc','E'] else ''
print("\n\t\t==== Model created  =====")
checkpoint = torch.load(checkpointname+'.pth.tar',map_location=torch.device(device)) if mn in ['BERT_spc','E'] else ''
print("\n\t\t==== Checkpoint loaded  =====")
print(mymodelST.load_state_dict(checkpoint['state_dict'])) if mn in ['BERT_spc','E'] else ''
del checkpoint
print("\n\n\t\t=========== MyModel loaded ===========")
if mn == 'BERT_sim':
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	# if "mbert" in suff:
	# 	bert_model_name = "bert-base-multilingual-cased"
	# elif "indic" in suff:
	# 	bert_model_name = "ai4bharat/indic-bert"
	# elif "muril" in suff:
	# 	bert_model_name = "simran-kh/muril-cased-temp"
	# else:
	# 	raise "this error"
	mytokenizer = AutoTokenizer.from_pretrained(bert_model_name)
	SimModel = SiameseNetwork(emb_opt=2,res_opt=4,bert_model_name=bert_model_name,device=device,sig=0).to(device)
	print("\n\n\t\t=========== Siamese loaded ===========")
	mpaths = {
		"_mbert_lhsAvg_cos_nosig_fullDataset_ep1":"model-exp_id-0-dated-Fri Aug  6 07:00:11 2021-best.pth",
		"_mbert_pool_add_nosig":"model-exp_id-0-dated-Fri Aug  6 06:43:25 2021-best.pth",
		"_mbert_first_two_add_nosig":"model-exp_id-0-dated-Fri Aug  6 06:48:07 2021-best.pth",
		"_mbert_cls_add_nosig":"model-exp_id-0-dated-Fri Aug  6 06:51:11 2021-best.pth",
		"_mbert_pool_cat_nosig":"model-exp_id-0-dated-Tue Aug  3 21:54:18 2021-best.pth",
		"_mbert_first_two_cat_nosig":"model-exp_id-0-dated-Tue Aug  3 21:51:13 2021-best.pth",
		"_mbert_cls_cat_nosig":"model-exp_id-0-dated-Tue Aug  3 21:46:46 2021-best.pth",
		"_mbert_lhsAvg_cos_nosig_final":"model-exp_id--2-epoch-5-dated-Tue Aug  3 01:17:55 2021.pth",
		"_mbert_lhsAvg_cos_nosig_best":"model-exp_id--2-dated-Tue Aug  3 01:17:55 2021-best.pth",
		"_mbert_uncased_lhsAvg_cos_nosig":"model-exp_id--1-dated-Mon Aug  2 23:34:35 2021-best.pth",
		"_muril_lhsAvg_cos_nosig":"model-exp_id--1-dated-Mon Aug  2 23:24:10 2021-best.pth",
		"_mbert_cased_lhsAvg_cos_nosig":"model-exp_id--1-dated-Mon Aug  2 06:29:17 2021-best.pth",
		"_albert_lhsAvg_cos_nosig":"model-exp_id--1-dated-Mon Aug  2 23:21:25 2021-best.pth",
		"_mbert_lhsAvg_cat_nosig":"model-exp_id--1-dated-Mon Aug  2 06:36:33 2021-best.pth",
		"_mbert_lhsAvg_cos_nosig":"model-exp_id--1-dated-Mon Aug  2 06:29:17 2021-best.pth",
		"_mbert_lhsAvg_add_nosig":"model-exp_id--1-dated-Mon Aug  2 06:25:56 2021-best.pth",
		"_mbert_lhsAvg_cat":"model-exp_id--1-dated-Sun Aug  1 16:40:46 2021-best.pth",
		"_mbert_lhsAvg_add":"model-exp_id--1-dated-Sun Aug  1 16:37:28 2021-best.pth",
		"_mbert_last_two_cosine":"model-exp_id--1-dated-Sun Aug  1 15:08:06 2021-best.pth",
		"_mbert_first_two_cosine":"model-exp_id--1-dated-Sun Aug  1 15:04:46 2021-best.pth",
		"_mbert_pool_cosine":"model-exp_id--1-dated-Sun Aug  1 14:59:23 2021-best.pth",
		"_mbert_lhsAvg_cosine":"model-exp_id--1-dated-Sun Aug  1 14:50:30 2021-best.pth",
		"_mbert_cls_cosine":"model-exp_id--1-dated-Sun Aug  1 14:43:15 2021-best.pth",
		"_mbert_exp6_epoch_last":"model-exp_id-6-epoch-5-dated-Mon Jul 26 07:42:05 2021.pth",
		"_mbert_exp6_epoch0":"model-exp_id-6-epoch-1-dated-Mon Jul 26 07:42:05 2021.pth",
		"_mbert_rand_neg":"model-exp_id-6-dated-Fri Jul 16 16:15:50 2021-best.pth", 
		"_mbert_33k":"model-exp_id-6-dated-Tue Jul 13 01:12:31 2021-best.pth", 
		"_indic_bert_rand_neg":"model-exp_id-6-dated-Thu Jul 15 23:33:05 2021-best.pth", 
		"_indic_bert_33k":"model-exp_id-6-dated-Fri Jul 16 15:00:07 2021-best.pth",
		"_muril_rand_neg":"model-exp_id-6-dated-Fri Jul 16 17:39:58 2021-best.pth",
		"_muril_33k":"model-exp_id-6-dated-Tue Jul 13 02:46:03 2021-best.pth",
		"_mbert_df.h5":"model-exp_id-6-dated-Wed Jul 21 02:33:58 2021-best.pth"}
	print(SimModel.load_state_dict(torch.load(mpaths[suff])))
	SimModel.eval()
else:
	SimModel = ''
print('\n\n\t\t=========== SimModel loaded ===========')

allQnA =  torch.load('allQnA.bin') if mn in ['BERT_cos', 'E'] else pd.read_csv('allQnA.csv') # path to your bin file
print("\n\n\t\t=========== Embedding file loaded ===========")

statnow(av, a)
# input('wait')
# exit()

# allQnA['c_lemmaQuestion'] = allQnA['c_lemmaQuestion'].apply(lambda d: add_word_groups(d))
# allQnA['lemmaQuestion'] = allQnA['lemmaQuestion'].apply(lambda d: add_word_groups(d))
# allQnA['Question_a'] = allQnA['Question'].apply(lambda d: add_word_groups(clean(d))) # when BERT_spc_added_words in on
# allQnA['embeddings'] = torch.tensor(modelST.encode(allQnA['Question_a']).tolist())
# allQnA['k_lemmaQuestion'] = allQnA['lemmaQuestion'].progress_apply(lambda d: key_words(d))


if query_expansion:
	allQnA['Question_clean'] = allQnA['Question'].apply(lambda d: add_word_groups(clean(d)))
else:
	allQnA['Question_clean'] = allQnA['Question'].apply(lambda d: clean(d))
allQnA['Answer_clean'] = allQnA['Answer'].apply(lambda d: clean(d))
# allQnA['embeddings'] = torch.tensor(modelST.encode(allQnA['Question_clean']).tolist())

# now = time.time()
# print(mymodelST.predict("दूध पिलाने से सूज जाती हो तो क्या करे ","दूध पिलाने से सूज जाती हो तो क्या करे "), time.time()-now, 'secs') # 0.23 secs
# exit()

inout = 'Outside'
asked_qs = pd.read_csv(inout+'_questions.csv')
asked_qs['User_query'] = asked_qs['User_query'].apply(lambda d: clean(d))
GT = pd.read_csv(inout+'_questions_ground_truth_plus_trimurti.csv')
GT['User_query'] = GT['User_query'].apply(lambda d: clean(d))
GT['Database_question'] = GT['Database_question'].apply(lambda d: clean(d))


GT = GT[GT['Final score']!=0].reset_index(drop=True)
ml = list(set(asked_qs['User_query']) - set(GT['User_query'])) # inke answers ni hai db me
outdata = asked_qs[asked_qs['User_query'].apply(lambda d: d not in ml)].reset_index(drop=True) # inke answers hai db me, I didn't take directly from GT because usme categories ni thi

colname = 'User_query' #'Question'
fname = inout+'_questions' #'topic_wise'

# outdata = pd.read_csv(fname+'.csv')
# outdata = outdata[outdata[colname].notnull()].reset_index(drop=True)
# outdata = outdata.drop(columns=['Question'])

for i in tqdm(range(len(outdata)), desc='Running user queries'):
	q = outdata.iloc[i][colname]
	q = clean(q)
	# q = trn.transform(q)

	if query_expansion:
		q = add_word_groups(q)
	# q = key_words(q)

	q2 = mn+' '+q
	# q2 = 'Ensemble '+q

	# result = getTopN(allQnA, q2, nlp, modelST, mymodelST, SimModel, show=False).iloc[:3].reset_index(drop=True)
	result = getTopN(allQnA, q2, nlp, modelST, mymodelST, SimModel, show=False)
	# result = pd.concat([result[result['category'] == outdata.loc[i,'Category']],result[result['category'] != outdata.loc[i,'Category']]]) # Oracle for categories
	result = result.reset_index(drop=True).iloc[:3] # top-k 

	for j in range(len(result)):
		outdata.at[i,'Model_name'] = mn+suff
		outdata.at[i,'OPTION_'+str(j+1)] = result.iloc[j]['Question']
		outdata.at[i,'ANSWER_'+str(j+1)] = result.iloc[j]['Answer']
		outdata.at[i,'SCORE_'+str(j+1)] = result.iloc[j]['score']
		
		fs = 3
		tdf = GT[GT['User_query'] == clean(outdata.iloc[i][colname])] # remember that the column here and the other string, both are clean. That is intentional.
		if len(tdf) > 0:
			tdf2 = tdf[tdf['Database_question'] == clean(result.iloc[j]['Question'])] # remember that the column here and the other string, both are clean. That is intentional.
			if len(tdf2) > 0:
				fs = 1 if max(list(tdf['Final score'])) == '2' else 2
		else:
			raise "ajeeb error, this question is not found in GT ->"+outdata.iloc[i][colname]+"<-"

		outdata.at[i,'final_'+str(j+1)] = fs
		

print(outdata)
outdata.to_csv(fname+'_'+mn+'_results.csv', index=None)
print(fname+'_'+mn+'_results.csv')
print(checkpointname)
print('='*100)



