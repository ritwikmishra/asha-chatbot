'''
this is for fine-tuning the BERT_spc
this also generates the gradient flow csvs
    and caculates SR on each fine-tuning iteration

data_source aur include_asha_data me change kiye hai (dictionary me)
aur hold_out testing off kari hai

'''

import datetime
import pytz, torch, json, sys
from utils import add_word_groups
from sklearn.metrics import f1_score
import argparse

IST = pytz.timezone('Asia/Kolkata')
stamp = datetime.datetime.now(IST).strftime("%c")


infoDict = {
    'devInfo': '',
    'dataLoaderInfo': '',
    'modelInfo': '',
    'optimInfo':''
}
# createData = True
# bs = 32 # BATCH_SIZE
# BERT_FROZEN = True # not used in BertSequenceClassification
# bert_model_name = 'bert-base-multilingual-cased' # 'ai4bharat/indic-bert' 'bert-base-multilingual-cased' 'xlm-roberta-base' 'bert-base-multilingual-uncased' 'xlm-mlm-100-1280' (out of memory error) 'xlm-mlm-tlm-xnli15-1024' 'xlm-mlm-xnli15-1024' (poor)
# alpha = 0.00001 #0.0001
# epochs = 5
# rseed = 123
# nw = 2
# cut_off = 0.51
# label_smoothing = False
# random_negative_sent = False
# train_ratio =  0.5
# val_ratio = 0.2
# max_len = 300 #200 #100
# data_source = 'inshorts_400pairs' # 'asha_sentsim', 'inshorts_full', 'inshorts_400pairs'
# architecture = 'SequenceClassification'  # 'SequenceClassification', 'last_two', first_two
# run_ID = 96 #int(torch.load(myfilepath+'run_ID.bin')) + 1

parser = argparse.ArgumentParser(description="arguments here")
parser.add_argument("--rid", default="0")
parser.add_argument("--log_file", default="bert_spc_logs_2.txt")
parser.add_argument("--message", default="")
parser.add_argument("--bs", default="32")
parser.add_argument("--bert_model_name", default="bert-base-multilingual-cased")
parser.add_argument("--alpha", default="1e-5")
parser.add_argument("--epochs", default="3")
parser.add_argument("--rseed", default="131")
parser.add_argument("--bert_frozen", default="embedding, layer0")
parser.add_argument("--rand_neg", default="1")
parser.add_argument("--data_source", default="inshorts_full")
parser.add_argument("--include_asha_data", default="after inshorts data")
parser.add_argument("--arch", default="fc3")
parser.add_argument("--which_embed", default="pooler_output")
parser.add_argument("--criterion", default="BCE")
parser.add_argument("--inflate_neg_n_times", default="0")
parser.add_argument("--include_curated_negs", default="0")
args = parser.parse_args()

hyper_params = {
    'run_ID': int(args.rid),
    'message': str(args.message), #'108 with a different seed',
	'createData' : True,
	'bs' : int(args.bs),
	'BERT_FROZEN' : str(args.bert_frozen), # not used in BertSequenceClassification
	'bert_model_name' : str(args.bert_model_name),
	'model_options': ', '.join(['ai4bharat/indic-bert', 'bert-base-multilingual-cased', 'xlm-roberta-base', 'bert-base-multilingual-uncased', 'xlm-mlm-100-1280', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024']),
	'alpha' : float(args.alpha), #1e-5,
	'epochs' : int(args.epochs),
	'rseed' : int(args.rseed),
	'nw' : 4,
	'cut_off' : 0.51,
	'label_smoothing' : False,
	'random_negative_sent' : bool(int(args.rand_neg)),
	'train_ratio' :  0.5,
	'val_ratio' : 0.2,
	'max_len' : 300,
	'data_source' : str(args.data_source), #'inshorts_full',
    'include_asha_data': str(args.include_asha_data), #str(sys.argv[14]),
	'data_options': ', '.join(['asha_sentsim', 'inshorts_full', 'inshorts_400pairs']),
	'architecture' : str(args.arch), #'fc2',
	'arch_options': ', '.join(['SequenceClassification', 'fc1', 'fc2','fc3']),
    'which_bert_embedding': str(args.which_embed),#'pooler_output',
    'bert_embedding_options': ', '.join(['cls','pooler_output']),
    'my_datapath': 'data/', 
    'criterion': str(args.criterion),#'HingeEmbeddingLoss',
    'query_expansion': False,
    'inflate_negative_n_times': int(args.inflate_neg_n_times),
    'include_curated_negatives': bool(int(args.include_curated_negs))
}
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
which_criterion = {
    'MSE': torch.nn.MSELoss(),
    'NLL': torch.nn.NLLLoss(),
    'CrossEntropy': torch.nn.CrossEntropyLoss(),
    'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss(),
    'BCE': torch.nn.BCELoss(),
    'BCEwithLogits': torch.nn.BCEWithLogitsLoss()
}

if hyper_params['include_asha_data'] == 'only asha data':
    hyper_params['data_source'] = 'asha_sentsim'

model_embeddings = {
    'ai4bharat/indic-bert':768,
    'bert-base-multilingual-cased':768,
    'xlm-roberta-base':768,
    'bert-base-multilingual-uncased':768,
    'xlm-mlm-100-1280':1280,
    'xlm-mlm-tlm-xnli15-1024':1024,
    'xlm-mlm-xnli15-1024':1024
}
if hyper_params['criterion'] == 'HingeEmbeddingLoss':
    hyper_params['cut_off'] = 0.0

hyper_params['embedding_size'] = model_embeddings[hyper_params['bert_model_name']]
# mydatapath, myfilepath = ("gdrive/My Drive/SentSimHi/", "gdrive/My Drive/SentSimHi/") if input('Google Colab? Enter anything\n2 Midas? Press Enter ') else ("../gdrive/My Drive/SentSimHi/", "")
mydatapath, myfilepath = ("data/", "")
logFile = myfilepath+str(args.log_file) #'bert_spc_logs_2.txt'

import os, torch, random, numpy as np
os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
# Torch RNG
torch.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed_all(hyper_params['rseed'])
# Python RNG
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])
from transformers import set_seed
set_seed(hyper_params['rseed'])

def plot_grad_flow(named_parameters, message='', save=True, epoch=0, iter=0, hyper_params=hyper_params):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    l2_norm_grads = []
    layers = []
    run_ID = hyper_params['run_ID']
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(float(p.grad.abs().mean()))
            max_grads.append(float(p.grad.abs().max()))
            l2_norm_grads.append(float(torch.norm(p.grad,2)))
    try:
        df = pd.read_csv(hyper_params['my_datapath']+'GradData_run_ID_'+str(run_ID)+'.csv')
    except:
        df = pd.DataFrame([],columns=['Epoch','Iteration','type']+layers)
    df = pd.concat([df,pd.DataFrame([[epoch,iter,'abs_avg']+ave_grads],columns=['Epoch','Iteration','type']+layers)]).reset_index(drop=True)
    df = pd.concat([df,pd.DataFrame([[epoch,iter,'abs_max']+max_grads],columns=['Epoch','Iteration','type']+layers)]).reset_index(drop=True)
    df = pd.concat([df,pd.DataFrame([[epoch,iter,'l2_norm']+l2_norm_grads],columns=['Epoch','Iteration','type']+layers)]).reset_index(drop=True)

    df.to_csv(hyper_params['my_datapath']+'GradData_run_ID_'+str(run_ID)+'.csv',index=None)
    return

# export CUDA_DEVICE_ORDER=PCI_BUS_ID # this works
import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda:0")
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = t -(c-a)  # free inside cache
    print("GPU is available", torch.cuda.get_device_name(), round(t/(1024*1024*1024)), "GB")
    infoDict['devInfo'] = "GPU is available "+str(torch.cuda.get_device_name())+' '+str(round(t/(1024*1024*1024)))+" GB"
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    infoDict['devInfo'] = "GPU not available, CPU used"


import glob
import pandas as pd
import random
import torch

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


# DATASET CLASS
# !pip install --upgrade tables > temp.txt
import random
import pandas as pd
from torch.utils import data
import torch
from transformers import BertTokenizer
from IPython.display import display, HTML
from transformers import BertTokenizer, XLMRobertaTokenizer, AutoTokenizer, XLMTokenizer


class mydataset(data.Dataset):
    def __init__(self, bert_model_name, name, max_len, data_source, seed=hyper_params['rseed']):
        if data_source == 'asha_sentsim':
            df2 = pd.read_csv('asha_sentsim_random_negs.csv')
        elif data_source == 'inshorts+dpil':
            if name != 'test':
                df2 = pd.read_hdf(hyper_params['my_datapath']+'sentence_pairs_35K.h5','ids')
                df3 = pd.read_csv('dpil_train_hi.csv')
                df3 = df3.rename(columns={'Sent1':'s1','Sent2':'s2','y':'score'})
                df3['score'] = df3['score'].apply(lambda d: 0.0 if d=='NP' else 1.0)
                assert sum(1.0 == df3['score'])+sum(0.0 == df3['score']) > 0
                df3 = df3[['s1','s2','score']]
                df2 = df2.append(df3).reset_index(drop=True)
            else:
                df3 = pd.read_csv('dpil_test_hi.csv')
                df3 = df3.rename(columns={'Sent1':'s1','Sent2':'s2','y':'score'})
                df3['score'] = df3['score'].apply(lambda d: 0.0 if d=='NP' else 1.0)
                assert sum(1.0 == df3['score'])+sum(0.0 == df3['score']) > 0
                df2 = df3[['s1','s2','score']]
        elif data_source == 'dpil': # or name=='test': # change this: remove name== wali condition
            if name != 'test':
                df3 = pd.read_csv('dpil_train_hi.csv')
                df3 = df3.rename(columns={'Sent1':'s1','Sent2':'s2','y':'score'})
                df3['score'] = df3['score'].apply(lambda d: 0.0 if d=='NP' else 1.0)
                assert sum(1.0 == df3['score'])+sum(0.0 == df3['score']) > 0
                df2 = df3[['s1','s2','score']]
            else:
                df3 = pd.read_csv('dpil_test_hi.csv')
                df3 = df3.rename(columns={'Sent1':'s1','Sent2':'s2','y':'score'})
                df3['score'] = df3['score'].apply(lambda d: 0.0 if d=='NP' else 1.0)
                assert sum(1.0 == df3['score'])+sum(0.0 == df3['score']) > 0
                df2 = df3[['s1','s2','score']]
        else:
            df2 = pd.read_hdf(hyper_params['my_datapath']+'sentence_pairs_35K.h5','ids') #original
        train_ratio = hyper_params['train_ratio']
        val_ratio = hyper_params['val_ratio']


        # --------- just to repeat experiment id 19 ----------
        # df1 = pd.read_hdf('data/df.h5','ids')
        # df1['score'] = 1.0
        # df2 = pd.DataFrame([],columns=['fID', 's1', 's2', 'score'])
        # df2 = pd.concat([df2,df1[['fID','Sent','Positive','score']].rename(columns={'Sent':'s1','Positive':'s2'})])
        # df1['score'] = 0.0
        # df2 = pd.concat([df2,df1[['fID','Sent','Negative','score']].rename(columns={'Sent':'s1','Negative':'s2'})]).reset_index(drop=True)
        # --------- ------------------------------- ----------
        if 'with inshorts data' in hyper_params['include_asha_data']:
            df_asha = pd.read_csv('asha_sentsim_random_negs.csv')
            df2 = df2.append(df_asha).reset_index(drop=True)

        # -------------------------------------------------------------------------
        # this part makes sure that sentence pairs are kept separately between train, val, and test
        df2 = df2.sort_values('s1').reset_index(drop=True)
        # print(data_source)
        # input('wait')
        if name == 'train':
            if data_source in ['asha_sentsim','inshorts_full','inshorts then dpil']:
                df2 = df2.iloc[:int(train_ratio*len(df2))].reset_index(drop=True)
            elif data_source in ['inshorts+dpil','dpil']:
                df2 = df2.iloc[:int((train_ratio+val_ratio)*len(df2))].reset_index(drop=True)
            else:
                df2 = df2.iloc[:2*int(400)].reset_index(drop=True)
            # df2.to_csv(hyper_params['my_datapath']+'data/'+name+'_asha.csv')
        elif name == 'val':
            if data_source in ['asha_sentsim','inshorts_full','inshorts then dpil']:
                df2 = df2.iloc[int(train_ratio*len(df2)):int((train_ratio+val_ratio)*len(df2))].reset_index(drop=True)
            elif data_source in ['inshorts+dpil','dpil']:
                df2 = df2.iloc[int((train_ratio+val_ratio)*len(df2)):].reset_index(drop=True)
            else:
                df2 = df2.iloc[2*int(400):2*int(600)].reset_index(drop=True)
            # df2.to_csv(hyper_params['my_datapath']+'data/'+name+'_asha.csv')
        else:
            if data_source in ['asha_sentsim','inshorts then dpil','inshorts_full']: # change this: add inshorts_full here
                df2 = df2.iloc[int((train_ratio+val_ratio)*len(df2)):].reset_index(drop=True)
            elif data_source in ['inshorts+dpil','dpil']: # change this: remove inshorts_full here
                pass 
            else:
                df2 = df2.iloc[2*int(600):2*int(800)].reset_index(drop=True)
            # df2.to_csv(hyper_params['my_datapath']+'data/'+name+'_asha.csv')
        # -------------------------------------------------------------------------

        orig_df2 = df2.copy()
        for inn in range(hyper_params['inflate_negative_n_times']):
            df2 = df2.append(orig_df2[orig_df2['score']==0]).reset_index(drop=True)


        if hyper_params['random_negative_sent'] and (('dpil' not in hyper_params['data_source']) or name!='test'): # change this: remove inshorts wala check
            df2.at[df2['score']==0,'s2'] = list((df2[df2['score']==0]['s2']).sample(frac=1, random_state=seed+1))
            if hyper_params['include_curated_negatives']:
                df2 = df2.append(orig_df2[orig_df2['score']==0]).reset_index(drop=True)
        

            
        df2 = df2.sample(frac=1, random_state=seed).reset_index(drop=True)
        s1 = list(df2['s1'])
        s2 = list(df2['s2'])
        try:
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name,local_files_only=True)
        except:
            print('Error occured while loading the tokenizer. Waiting for 30 seconds now.')
            time.sleep(30)
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        encoding = tokenizer(list(s1), list(s2), return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
        self.input_ids = encoding['input_ids']
        if 'token_type_ids' in encoding.keys():
            self.token_type_ids = encoding['token_type_ids']
        else:
            self.token_type_ids = encoding['input_ids']*0
        self.attention_mask = encoding['attention_mask']
        if hyper_params['criterion'] == 'HingeEmbeddingLoss':
            df2['score'] = df2['score'].apply(lambda d: -1.0 if int(d)==0 else 1.0)
        self.y = torch.tensor(list(df2['score'])).type('torch.FloatTensor')
    
    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.y[index]

    def __len__(self):
        return len(self.y)


from torch.utils import data

print('Data Loader building...')

if (hyper_params['createData']):
    av, a = stathere()
    # dataset = mydataset()
    bs = hyper_params['bs']
    nw = hyper_params['nw']
    train_set, val_set, test_set = mydataset(hyper_params['bert_model_name'],'train', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'val', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'test', hyper_params['max_len'], hyper_params['data_source']) #torch.utils.data.random_split(dataset,[int(len(dataset)*train_ratio), int(len(dataset)*val_ratio), len(dataset)-(int(len(dataset)*train_ratio) + int(len(dataset)*val_ratio))])
    train_data = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_data = data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=nw)
    test_data = data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=nw)
    print('Train set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data)))
    infoDict['dataLoaderInfo'] = 'Batch_size: '+str(bs)+'\nTrain set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data))+'\n'
    # torch.save(train_data,hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_train_data.bin')
    # torch.save(val_data,hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_val_data.bin')
    # torch.save(test_data,hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_test_data.bin')
    infoDict['dataLoaderInfo'] += statnow(av, a)
    infoDict['dataLoaderInfo'] += ('\nExample: ... '+ str(train_set[0][3]) + str(train_set[1][3]) + str(train_set[2][3]) + str(train_set[3][3]) + str(train_set[4][3]) + str(train_set[5][3])  + '...\n')
    # exit()

else:
    av, a = stathere()
    bs = hyper_params['bs']
    train_data = torch.load(hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_train_data.bin')
    val_data = torch.load(hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_val_data.bin')
    test_data = torch.load(hyper_params['my_datapath']+str(hyper_params['run_ID'])+'_test_data.bin')
    print('Batch_size: '+str(bs)+'\nOld sets\n'+'Old batches\n')
    infoDict['dataLoaderInfo'] = 'Batch_size: '+str(bs)+'\nOld sets\n'+'Old batches\n'
    infoDict['dataLoaderInfo'] += statnow(av, a)

import string, re
def clean(hi_str):
	puncts = string.punctuation+"‟“’❝❞‚‘‛❛❜❟"
	trans_string = hi_str.maketrans("१२३४५६७८९०","1234567890")
	hi_str = hi_str.translate(trans_string)
	hi_str = re.sub(r'[a-z]+','',hi_str.lower())
	hi_str = hi_str.translate(str.maketrans(puncts,' '*len(puncts))).lower().replace('|','').replace('।','')
	hi_str = re.sub(r'\s+',' ',hi_str).strip()
	return hi_str

# convert the to binary matrix 
def get_data(acolumns,right_ans):
    repd = []
    for _,vec in acolumns.iterrows(): # vec = [2,3,1]
        rep = []
        for ind in vec.values:
            if ind in right_ans:
                rep.append(1)
            else:
                rep.append(0)
        repd.append(rep)
    return np.array(repd)

def precision_at_k(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    prec = []
    for vec in data:
        rev_doc = 0
        tot_doc = 0
        prec_vec = []
        for doc in vec:
            if doc == 1:
                rev_doc +=1
            tot_doc +=1
            prec_vec.append(rev_doc/tot_doc)
        prec.append(np.array(prec_vec))
    return np.array(prec)

def reciprocal_rank(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    rr = []
    for vec in data: #  vec = [0,1,1]
        i = 1
        for ind in vec:
            if ind == 1:
                break
            i+=1
        #print(i)
        if i !=4:
            rr.append(1/i)
        else:
            rr.append(0)
    return np.array(rr)

def success_ratio(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    sr_mat = []
    for vec in data:
        #sr_mat.append(sum(vec)/len(vec))
        sr_mat.append(max(vec))
    
    return np.array(sr_mat)

def invert_rel(element):
    rmap = {1:2,2:1,3:0}
    return rmap[element] 

def precision_at_3(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    prec_at_k = precision_at_k(data)
    k = 3-1
    prec_at_3 = [vec[k] for vec in prec_at_k]
    return np.array(prec_at_3)

def avg_precision_at_3(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    data = precision_at_k(data)
    aprec = []
    for vec in data:
        aprec.append(sum(vec)/3)
    return np.array(aprec)

def ndcg(acolumn):
    """
    param : acolumn
    pandas series representing ans in each cell
    """
    idcg_vec = np.array([1/np.log2(2) , 1/np.log2(3) , 1/np.log2(4)])
    temp = pd.DataFrame()
    temp["final_1"] = acolumn["final_1"].apply(invert_rel)
    temp["final_2"] = acolumn["final_2"].apply(invert_rel)
    temp["final_3"] = acolumn["final_3"].apply(invert_rel)

    ndcg_vec = []

    for _,vec in temp.iterrows(): #  vec = [1,2,1] 
        vec = vec.values  
        idcg = sum(idcg_vec*sorted(vec,reverse=True))
        if idcg!=0:
            ndcg_vec.append(np.sum(vec*idcg_vec)/idcg)
        else:
            ndcg_vec.append(0)
    return np.array(ndcg_vec)

def grp_mean(grp_mat):
    return (np.mean(grp_mat[0]),np.mean(grp_mat[1]),np.mean(grp_mat[2]))

def calc_metrics(df1):
    """
    param : acolumn
    pandas series representing ans in each cell
    """
    acolumn = df1[["final_1","final_2","final_3"]] 
    data = get_data(acolumn,[1,2])
    m_mat = {}

    m_mat["map"] = np.mean(avg_precision_at_3(data))
    m_mat["mrr"] = np.mean(reciprocal_rank(data))
    m_mat["avg_sr"] = np.mean(success_ratio(data))
    m_mat["avg_ndcg"] = np.mean(ndcg(acolumn))
    m_mat["avg_p3"] = np.mean(precision_at_3(data))
    return m_mat

allQnA = pd.read_csv('allQnA.csv')
allQnA['Question_clean'] = allQnA['Question'].apply(lambda d: clean(d))
allQnA['Answer_clean'] = allQnA['Answer'].apply(lambda d: clean(d))
if hyper_params['query_expansion']:
    allQnA['Question_clean'] = allQnA['Question_clean'].apply(lambda d: add_word_groups(d))

# !pip install transformers > temp.txt
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class myBert(torch.nn.Module):
    def __init__(self,d,hyper_params,which_criterion):
        super(myBert, self).__init__()
        self.bert_model_name = hyper_params['bert_model_name']
        self.label_smoothing = hyper_params['label_smoothing']
        self.max_len = hyper_params['max_len']
        self.arch = hyper_params['architecture']
        self.hyper_params = hyper_params

        try:
            self.mytokenizer_for_hold_out = AutoTokenizer.from_pretrained(self.bert_model_name)
        except:
            time.sleep(30)
            self.mytokenizer_for_hold_out = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.best_sr = 0
        self.best_val_acc = 0
        self.cut_off = hyper_params['cut_off']

        if self.arch == 'SequenceClassification':
            self.model = AutoModelForSequenceClassification.from_pretrained(self.bert_model_name, return_dict=True, num_labels=1)

        else:
            self.model = AutoModel.from_pretrained(self.bert_model_name, return_dict=True, output_hidden_states=True)
            if self.arch == 'fc1':
                self.fc1 = nn.Linear(hyper_params['embedding_size'], 1)
                if hyper_params['criterion']=='HingeEmbeddingLoss':
                    self.activation1 = nn.Tanh()
                elif hyper_params['criterion']=='BCEwithLogits':
                    self.activation1 = nn.Identity()
                else:
                    self.activation1 = nn.Sigmoid()
            elif self.arch == 'fc2':
                self.fc1 = nn.Linear(hyper_params['embedding_size'], 400)
                self.activation1 = nn.ReLU()
                self.fc2 = nn.Linear(400,1)
                if hyper_params['criterion']=='HingeEmbeddingLoss':
                    self.activation1 = nn.Tanh()
                    self.activation2 = nn.Tanh()
                elif hyper_params['criterion']=='BCEwithLogits':
                    self.activation2 = nn.Identity()
                else:
                    self.activation2 = nn.Sigmoid()
            elif self.arch == 'fc3':
                self.fc1 = nn.Linear(hyper_params['embedding_size'], 400)
                self.activation1 = nn.ReLU()
                self.fc2 = nn.Linear(400,100)
                self.activation2 = nn.ReLU()
                self.fc3 = nn.Linear(100,1)
                if hyper_params['criterion']=='HingeEmbeddingLoss':
                    self.activation1 = nn.Tanh()
                    self.activation2 = nn.Tanh()
                    self.activation3 = nn.Tanh()
                elif hyper_params['criterion']=='BCEwithLogits':
                    self.activation3 = nn.Identity()
                else:
                    self.activation3 = nn.Sigmoid()
            elif self.arch == 'fc4':
                self.fc1 = nn.Linear(hyper_params['embedding_size'], 500)
                self.activation1 = nn.ReLU()
                self.fc2 = nn.Linear(500,300)
                self.activation2 = nn.ReLU()
                self.fc3 = nn.Linear(300,100)
                self.activation3 = nn.ReLU()
                self.fc4 = nn.Linear(100,1)
                if hyper_params['criterion']=='HingeEmbeddingLoss':
                    self.activation1 = nn.Tanh()
                    self.activation2 = nn.Tanh()
                    self.activation3 = nn.Tanh()
                    self.activation4 = nn.Tanh()
                elif hyper_params['criterion']=='BCEwithLogits':
                    self.activation4 = nn.Identity()
                else:
                    self.activation4 = nn.Sigmoid()
            self.criterion = which_criterion[hyper_params['criterion']]

        # self.optim = AdamW(self.model.parameters(), lr=alpha)
        self.device = d

    def take_mean(self,tensorlist):
        out = 0
        for t in tensorlist:
            out += t
        out /= len(tensorlist)
        return out
    
    def fc_forward(self,out):
        if self.arch == 'last_two':
            return self.fc1(self.take_mean([ out.hidden_states[-2][:,0], out.hidden_states[-3][:,0] ]))
        elif self.arch == 'first_two':
            return self.fc1(self.take_mean([ out.hidden_states[0][:,0], out.hidden_states[0][:,0] ]))
        elif self.arch == 'fc1':
            out = self.fc1(out)
            out = self.activation1(out)
        elif self.arch == 'fc2':
            out = self.fc1(out)
            out = self.activation1(out)
            out = self.fc2(out)
            out = self.activation2(out)
        elif self.arch == 'fc3':
            out = self.fc1(out)
            out = self.activation1(out)
            out = self.fc2(out)
            out = self.activation2(out)
            out = self.fc3(out)
            out = self.activation3(out)
        elif self.arch == 'fc4':
            out = self.fc1(out)
            out = self.activation1(out)
            out = self.fc2(out)
            out = self.activation2(out)
            out = self.fc3(out)
            out = self.activation3(out)
            out = self.fc4(out)
            out = self.activation4(out)
        else:
            raise "Error: Unknown architecture"
        return out
    
    def forward(self, input_ids, token_type_ids, attention_mask, y):
      if self.label_smoothing:
        y = y + (torch.rand(len(y))/1000).to(self.device)
      
      if self.arch == 'SequenceClassification':
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=y.unsqueeze(0))
      else:
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.hyper_params['which_bert_embedding'] == 'pooler_output':
            out = out.pooler_output
        else:
            out = out.last_hidden_state[:,0]
        out = self.fc_forward(out)
        if self.hyper_params['criterion'] == 'HingeEmbeddingLoss':
            out = 1-out
        # print(out.shape, y.shape)
        # print(y.unsqueeze(1).shape)
        # print(out)
        # print(y)
        # print(y.unsqueeze(1))
        # input()
        loss = self.criterion(out, y.unsqueeze(1))
        out = SequenceClassifierOutput(loss=loss, logits=out)
      return out

    def forward_for_testing(self, input_ids, token_type_ids, attention_mask):
        if self.arch == 'SequenceClassification':
            out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if self.hyper_params['which_bert_embedding'] == 'pooler_output':
                out = out.pooler_output
            else:
                out = out.last_hidden_state[:,0]
            out = self.fc_forward(out)
            out = SequenceClassifierOutput(loss=0, logits=out)
        return out
    
    def hold_out_testing(self, run_ID, epoch, iter):
        inout = 'Outside'
        asked_qs = pd.read_csv(inout+'_questions.csv')
        asked_qs['User_query'] = asked_qs['User_query'].apply(lambda d: clean(d))
        GT = pd.read_csv(inout+'_questions_ground_truth_plus_trimurti.csv')
        GT['User_query'] = GT['User_query'].apply(lambda d: clean(d))
        GT['Database_question'] = GT['Database_question'].apply(lambda d: clean(d))

        GT = GT[GT['Final score']!=0].reset_index(drop=True)
        ml = list(set(asked_qs['User_query']) - set(GT['User_query'])) # inke answers ni hai db me
        outdata = asked_qs[asked_qs['User_query'].apply(lambda d: d not in ml)].reset_index(drop=True) # inke answers hai db me, I didn't take directly from GT because usme categories ni thi

        colname = 'User_query'

        for i in tqdm(range(len(outdata)), desc='Hold_out', ncols=100):
            q = outdata.iloc[i][colname]
            q = clean(q)
            if self.hyper_params['query_expansion']:
                q = add_word_groups(q)

            allQnA['score'] = self.databaseSearch(q,allQnA,self.mytokenizer_for_hold_out)

            result = allQnA[['category','Question','Answer','score']]
            result = result.sort_values(['score'], ascending=[False]).reset_index(drop=True)

            result = result.reset_index(drop=True).iloc[:3]

            for j in range(len(result)):
                outdata.at[i,'Model_name'] = 'any-name'
                outdata.at[i,'OPTION_'+str(j+1)] = result.iloc[j]['Question']
                outdata.at[i,'ANSWER_'+str(j+1)] = result.iloc[j]['Answer']
                outdata.at[i,'SCORE_'+str(j+1)] = result.iloc[j]['score']
                
                fs = 3 # final score
                tdf = GT[GT['User_query'] == clean(outdata.iloc[i][colname])] # remember that the column here and the other string, both are clean. That is intentional.
                if len(tdf) > 0:
                    tdf2 = tdf[tdf['Database_question'] == clean(result.iloc[j]['Question'])] # remember that the column here and the other string, both are clean. That is intentional.
                    if len(tdf2) > 0:
                        fs = 1 if max(list(tdf['Final score'])) == '2' else 2
                else:
                    raise "ajeeb error, this question is not found in GT ->"+outdata.iloc[i][colname]+"<-"

                outdata.at[i,'final_'+str(j+1)] = fs

        df = outdata
        df.to_csv(hyper_params['my_datapath']+'run_ID_'+str(run_ID)+'_epoch'+str(epoch)+'_hold_out_results.csv',index=None)
        md = calc_metrics(df)

        file = open(hyper_params['my_datapath']+'train_hold_out_metrics_'+str(self.hyper_params['run_ID'])+'.txt','a')
        file.write('Epoch_'+str(epoch)+' Iter='+str(iter)+' ')
        for k in md.keys():
            file.write(k+':'+str(md[k])+' ')
        file.write('\n')
        file.close()

        if md['avg_sr'] > self.best_sr:
            self.best_sr = md['avg_sr']
            # df.to_csv(hyper_params['my_datapath']+'run_ID_'+str(run_ID)+'_best_sr.csv',index=None)
            # torch.save({'state_dict': self.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(run_ID)+'_best_sr.pth.tar')

        return

    def start_training(self, e, run_ID, optim, train_data, val_data, test_data, logging=''):
        ttime = time.time()
        self.train()
        ep_train_loss = []

        for i, (input_ids, token_type_ids, attention_mask, y) in tqdm(enumerate(train_data), desc='training /'+str(len(train_data)),total=len(train_data), ncols=100):
            self.train()
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            y = y.to(device)

            optim.zero_grad()
            out = self.forward(input_ids,token_type_ids,attention_mask,y)
            # print('Logits', out.logits,out.logits.shape)
            # print('Y=', y, y.shape)
            # print('Loss:',out.loss)
            # input()

            tr_loss = out.loss
            tr_loss.backward()
            # plot_grad_flow(model.named_parameters(), "Epoch "+str(e)+"- Batch "+str(i), save=True, epoch=e, iter=i)
            ep_train_loss.append(tr_loss.item())
            optim.step()

            file = open(hyper_params['my_datapath']+'train_loss_'+str(self.hyper_params['run_ID'])+'.txt','a')
            file.write('Epoch_'+str(e)+' Iter='+str(i)+' '+str(tr_loss.item())+'\n')
            file.close()

            # if i > 10:
            #     break

            del out, input_ids, token_type_ids, attention_mask, y, tr_loss

        self.validation(e, run_ID, val_data, logging, iter=i, message='train')
        self.start_testing( run_ID, test_data, logging, e=e, iter=i, message='train')
        if e+1 == 4:
            self.hold_out_testing(run_ID, e, i)


        
        tttaken = time.time() - ttime
        ep_train_loss = round(sum(ep_train_loss)/len(ep_train_loss),3)
        print("\nEpoch:",e+1,"Train Loss", ep_train_loss, "Time taken:", round(tttaken, 2), 'secs')
        if logging:
            f = open(logging, 'a')
            f.write("\nEpoch: "+str(e+1)+" Train Loss "+str(ep_train_loss)+" Time taken: "+str(round(tttaken, 2))+' secs\n')
            f.close()
        return ep_train_loss

    def validation(self, e, run_ID, val_data, logging='', iter=0, message=''):
        ttime = time.time()
        self.eval()
        y_pred = []
        y_true = []

        ep_valid_loss = []

        for i, (input_ids, token_type_ids, attention_mask, y) in tqdm(enumerate(val_data), desc='validation /'+str(len(val_data)),total=len(val_data), ncols=100):
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            y = y.to(device)

            with torch.no_grad():
                out = self.forward(input_ids,token_type_ids,attention_mask,y)

            vd_loss = out.loss

            ep_valid_loss.append(vd_loss.item())
            # del out

            # out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.detach()
            out = out.logits
            if self.hyper_params['criterion'] == 'HingeEmbeddingLoss':
                out = 1 - out # this is done to reverse the operation that happens in 'forward' function

            out[out > self.cut_off] = 1.0
            out[out < self.cut_off] = 0.0

            out = out.squeeze().tolist()
            y[y==-1] = 0.0 # special case for hinge loss
            y = y.tolist()

            if type(out) != list:
                assert type(out) == float or type(out) == int
                out = [out]
            if type(y) != list:
                assert type(y) == float or type(y) == int
                y = [y]

            y_pred += out
            y_true += y
            
            del out, input_ids, token_type_ids, attention_mask, y, vd_loss

            # if i > 10:
            #    break

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)

        tttaken = time.time() - ttime
        ep_valid_loss = round(sum(ep_valid_loss)/len(ep_valid_loss),3)
        if self.hyper_params['inflate_negative_n_times'] > 0:
            ep_valid_acc = f1_score(y_true, y_pred, average='macro') #round(100*int(sum(y_true == y_pred))/len(y_true),3)
        else:
            ep_valid_acc = round(100*int(sum(y_true == y_pred))/len(y_true),3)

        if message == 'train':
            file = open(hyper_params['my_datapath']+'train_val_loss_acc_'+str(self.hyper_params['run_ID'])+'.txt','a')
            file.write('Epoch_'+str(e)+' Iter='+str(iter)+' '+str(ep_valid_loss)+' '+str(ep_valid_acc)+'\n')
            file.close()
        else:
            print("\n\tEpoch:",e+1,"Validation Loss", ep_valid_loss, "Time taken:", round(tttaken, 2), 'secs| ','Accuracy '+str(ep_valid_acc)+'\n')
            if logging:
                f = open(logging, 'a')
                f.write("\n\tEpoch: "+str(e+1)+" Validation Loss "+str(ep_valid_loss)+" Time taken: "+str(round(tttaken, 2))+' secs| ')
                if self.hyper_params['inflate_negative_n_times'] > 0:
                    msg = 'Macro F: '+str(round(ep_valid_acc,3))+'\n'
                else:
                    msg = 'Accuracy '+str(round(ep_valid_acc,3))+'\n'
                f.write(msg)
                f.close()
        return ep_valid_loss, ep_valid_acc

    def start_testing(self, run_ID, test_data, logging='', e = 0, iter=0, message=''):
        ttime = time.time()
        self.eval()
        y_pred = []
        y_true = []

        for i, (input_ids, token_type_ids, attention_mask, y) in tqdm(enumerate(test_data), desc='testing /'+str(len(test_data)), total=len(test_data), ncols=100):
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            y = y.to(device)

            with torch.no_grad():
                out = self.forward_for_testing(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

            out[out > self.cut_off] = 1.0
            out[out < self.cut_off] = 0.0

            out = out.squeeze().tolist()
            y[y==-1] = 0.0 # special case for hinge loss
            y = y.tolist()

            if type(out) != list:
                assert type(out) == float or type(out) == int
                out = [out]
            if type(y) != list:
                assert type(y) == float or type(y) == int
                y = [y]

            y_pred += out
            y_true += y

            # if i > 10:
            #    break


        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        if self.hyper_params['inflate_negative_n_times'] > 0:
            ep_test_acc = f1_score(y_true, y_pred, average='macro') #round(100*int(sum(y_true == y_pred))/len(y_true),3)
        else:
            ep_test_acc = round(100*int(sum(y_true == y_pred))/len(y_true),3)

        if message == 'train':
            file = open(hyper_params['my_datapath']+'train_test_acc_'+str(self.hyper_params['run_ID'])+'.txt','a')
            file.write('Epoch_'+str(e)+' Iter='+str(iter)+' '+str(ep_test_acc)+'\n')
            file.close()
        else:
            ts = 'T\\P\t\t 0\t\t 1\n'
            print('T\\P\t\t 0\t\t 1')
            b = 0
            for r in confusion_matrix(y_true, y_pred):
                print(str(b),end='\t\t')
                ts+=(str(b)+'\t\t')
                for c in r:
                    print(str(c).zfill(4), end='\t\t')
                    ts+=(str(c).zfill(4)+'\t\t')
                print('')
                ts+='\n'
                b+=1
            if self.hyper_params['inflate_negative_n_times'] > 0:
                msg = '\nMacro F: '+str(round(ep_test_acc,3))+'\n'
            else:
                msg = '\nAccuracy '+str(round(ep_test_acc,3))+'\n'
            print(msg)
            ts+=(msg)
            tttaken = time.time() - ttime
            print('Time taken:',round(tttaken, 2),'secs')
            ts+=('Time taken: '+str(round(tttaken, 2))+' secs\n')
            if logging:
                f = open(logging, 'a')
                f.write(ts)
                f.close()
        return

    def predict(self,sent1,sent2,tokenizer=''):
        self.eval()
        if tokenizer == '':
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            except:
                time.sleep(30)
                tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        encoding = tokenizer([sent1,''], [sent2,''], return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True).to(self.device)
        input_ids = encoding['input_ids']
        if 'token_type_ids' in encoding.keys():
            token_type_ids = encoding['token_type_ids']
        else:
            token_type_ids = encoding['input_ids']*0
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            out = self.forward_for_testing(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        out = out.to('cpu').detach().numpy()
        return out[0].item()

    # I made a loop in this function because without it 1K x 100 sized vector used to gave cuda out of memory error. I deleted the variable in each iteration.
    def databaseSearch(self, q, df, tokenizer):
        self.eval()
        output = torch.tensor([]).to(self.device)
        for i in (range(0,len(df),30)):
            df2 = df[i:i+30]
            # ----------------- Question block -----------------
            encoding = tokenizer([q]*(len(df2)), list(df2['Question_clean']), return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(self.device)
            input_ids = encoding['input_ids']
            if 'token_type_ids' in encoding.keys():
                token_type_ids = encoding['token_type_ids']
            else:
                token_type_ids = encoding['input_ids']*0
            attention_mask = encoding['attention_mask']
            with torch.no_grad():
                out = self.forward_for_testing(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.detach()
            del encoding
            # --------------------------------------------------


            # ----------------- Question + Answer block -----------------
            # encoding = tokenizer([q]*(len(df2)), list(df2['Question_clean']), return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(self.device)
            # input_ids = encoding['input_ids']
            # if 'token_type_ids' in encoding.keys():
            #     token_type_ids = encoding['token_type_ids']
            # else:
            #     token_type_ids = encoding['input_ids']*0
            # attention_mask = encoding['attention_mask']
            # with torch.no_grad():
            #     out1 = self.forward_for_testing(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.detach()
            # del encoding
            # encoding = tokenizer([q]*(len(df2)), list(df2['Answer_clean']), return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(self.device)
            # input_ids = encoding['input_ids']
            # if 'token_type_ids' in encoding.keys():
            #     token_type_ids = encoding['token_type_ids']
            # else:
            #     token_type_ids = encoding['input_ids']*0
            # attention_mask = encoding['attention_mask']
            # with torch.no_grad():
            #     out2 = self.forward_for_testing(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.detach()
            # del encoding
            # out = (out1+out2)/2
            # -----------------------------------------------------------
            output = torch.cat((output,out), dim=0)


        output = output.to('cpu').numpy()
        return output


# ------------- to temporarily test the bert output
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
# bert(input_ids=train_set[0][0].unsqueeze(0), token_type_ids=torch.tensor([0]*500).unsqueeze(0), attention_mask=train_set[0][2].unsqueeze(0))

av, a = stathere()
print('Creating the model...')

try:
    model = myBert(device,hyper_params,which_criterion).to(device)
except Exception as e:
    if 'Connection error' in str(e):
        time.sleep(30)
        model = myBert(device,hyper_params,which_criterion).to(device)

num_bertLayers = len(list(model.model.named_parameters()))
if hyper_params['BERT_FROZEN'] == '160':
    for pi,param in enumerate(model.model.parameters()): # freeze first half
        if pi < int(num_bertLayers/2):
            param.requires_grad = False
elif hyper_params['BERT_FROZEN'] == '163':
    for pi,param in enumerate(model.model.parameters()): # freeze first third matlab first 4 layers
        if pi < int(num_bertLayers/3):
            param.requires_grad = False
elif hyper_params['BERT_FROZEN'] == 'embedding, layer0':
    for pi,(name,param) in enumerate(model.model.named_parameters()): # freeze embedding, layer0,
        if 'encoder.layer.1' in name:
            break
        else:
            param.requires_grad = False
elif hyper_params['BERT_FROZEN'] == 'embedding, layer0, layer1':
    for pi,(name,param) in enumerate(model.model.named_parameters()): # freeze embedding, layer0, layer1 
        if 'encoder.layer.2' in name:
            break
        else:
            param.requires_grad = False
elif hyper_params['BERT_FROZEN'] == 'embedding, layer0, layer1, layer2':
    for pi,(name,param) in enumerate(model.model.named_parameters()): # freeze embedding, layer0, layer1, layer2 
        if 'encoder.layer.3' in name:
            break
        else:
            param.requires_grad = False
elif hyper_params['BERT_FROZEN'] == 'only embedding layer':
    for pi,(name,param) in enumerate(model.model.named_parameters()): # freeze embedding layer
        if 'encoder.layer.0' in name:
            break
        else:
            param.requires_grad = False
# infoDict['modelInfo'] = "full_inshorts | NEW CODE with activation = identity; fc_input = average of last 2 layers of BERT (excluding pooling_output): 62 repeat | SequenceClassification| train_ratio:"+str(train_ratio)+" val_ratio:"+str(val_ratio)+" | random negative-sentence: "+str(random_negative_sent)+"| model:"+bert_model_name+"\n"
# infoDict['modelInfo'] = "NEW CODE; fc_input = average of first 2 layers of BERT | optim separated | SequenceClassification | 400 sentence pairs for train, 200 for valid and test | random negative-sentence: "+str(random_negative_sent)+"| model:"+bert_model_name+"\n"
# infoDict['modelInfo'] = data_source+" | "+str(rseed)+" random seed | arch:"+architecture+" | random negative-sentence: "+str(random_negative_sent)+"| model:"+bert_model_name+"| max_len: "+str(max_len)+"\n"

infoDict['modelInfo'] = str(json.dumps(hyper_params, indent = 4))+'\n'+str(model)[:29]+' '+hyper_params['bert_model_name']+' )'+str(model)[13320:]+'\n'

infoDict['modelInfo'] += statnow(av,a)

optim = AdamW(model.parameters(), lr=hyper_params['alpha'])

infoDict['optimInfo'] = str(optim)


# TRAINING starts
import time
import datetime

print('run_ID is',hyper_params['run_ID'])

print('Training Starts...')

if logFile:
    f = open(logFile, 'a')
    f.write('\n\n'+'='*70+'\n'+'-'*70+'\n')
    f.write('\t\t\t\trun_ID='+str(hyper_params['run_ID'])+"\n")
    f.write(stamp+'\n')
    for k in infoDict.keys():
        f.write(infoDict[k]+'\n\n')
    f.close()
    

train_loss = []
val_loss = []


start_time = time.time()

for i in range(hyper_params['epochs']):
    print('-'*50,'Epoch',i+1,'-'*50)
    if hyper_params['BERT_FROZEN'] == '151':
        if i <= 1:
            for pi,param in enumerate(model.model.parameters()): # freeze first half
                if pi < int(num_bertLayers/2):
                    param.requires_grad = False
        else:
            for pi,param in enumerate(model.model.parameters()): # unfreeze first half
                if pi < int(num_bertLayers/2):
                    param.requires_grad = True
    elif hyper_params['BERT_FROZEN'] == '154':
        if i <= 1:
            for pi,param in enumerate(model.model.parameters()): # freeze second half
                if pi > int(num_bertLayers/2):
                    param.requires_grad = False
        else:
            for pi,param in enumerate(model.model.parameters()): # unfreeze second half
                if pi > int(num_bertLayers/2):
                    param.requires_grad = True
    elif hyper_params['BERT_FROZEN'] == '157':
        if i <= 1:
            for pi,param in enumerate(model.model.parameters()): # freeze first half
                if pi < int(num_bertLayers/2):
                    param.requires_grad = False
        elif i <= 3:
            for pi,param in enumerate(model.model.parameters()): # unfreeze first half and freeze second half
                if pi < int(num_bertLayers/2):
                    param.requires_grad = True
                if pi >= int(num_bertLayers/2):
                    param.requires_grad = False
        else:
            for pi,param in enumerate(model.model.parameters()): # unfreeze all
                param.requires_grad = True

    tloss = model.start_training(i, hyper_params['run_ID'], optim, train_data, val_data, test_data, logging=logFile)
    train_loss.append(tloss)
    vloss, vacc = model.validation(i, hyper_params['run_ID'], val_data, logging=logFile)
    val_loss.append(vloss)
    # if 'best model' in hyper_params['include_asha_data']:
    #     if vacc > model.best_val_acc:
    #         print("best model saved")
    #         torch.save({'state_dict': model.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_best.pth.tar')
    #         model.best_val_acc = vacc
    # torch.save({'state_dict': model.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_epoch_'+str(i)+'.pth.tar')

model.start_testing(hyper_params['run_ID'],test_data,logFile)

if 'after inshorts data' in hyper_params['include_asha_data']:
    hyper_params['data_source'] = 'asha_sentsim'
    train_set, val_set, test_set = mydataset(hyper_params['bert_model_name'],'train', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'val', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'test', hyper_params['max_len'], hyper_params['data_source']) #torch.utils.data.random_split(dataset,[int(len(dataset)*train_ratio), int(len(dataset)*val_ratio), len(dataset)-(int(len(dataset)*train_ratio) + int(len(dataset)*val_ratio))])
    train_data = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_data = data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=nw)
    test_data = data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=nw)
    print('Train set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data)))
    if logFile:
        f = open(logFile,'a')
        f.write('\nIncluding asha data now\n')
        f.write('Batch_size: '+str(bs)+'\nTrain set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data))+'\n')
        f.close()
    if 'with best model' in hyper_params['include_asha_data']:
        checkpoint = torch.load(hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_best.pth.tar')
        print(model.load_state_dict(checkpoint['state_dict']))
    for i in range(hyper_params['epochs'],hyper_params['epochs']+1):
        print('-'*50,'Epoch',i+1,'-'*50)
        tloss = model.start_training(i, hyper_params['run_ID'], optim, train_data, val_data, test_data, logging=logFile)
        train_loss.append(tloss)
        vloss, vacc = model.validation(i, hyper_params['run_ID'], val_data, logging=logFile)
        val_loss.append(vloss)
        # torch.save({'state_dict': model.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_epoch_'+str(i)+'.pth.tar')
    model.start_testing(hyper_params['run_ID'],test_data,logFile)

if 'inshorts then dpil' in hyper_params['data_source']:
    hyper_params['data_source'] = 'dpil'
    train_set, val_set, test_set = mydataset(hyper_params['bert_model_name'],'train', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'val', hyper_params['max_len'], hyper_params['data_source']), mydataset(hyper_params['bert_model_name'],'test', hyper_params['max_len'], hyper_params['data_source']) #torch.utils.data.random_split(dataset,[int(len(dataset)*train_ratio), int(len(dataset)*val_ratio), len(dataset)-(int(len(dataset)*train_ratio) + int(len(dataset)*val_ratio))])
    train_data = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_data = data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=nw)
    test_data = data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=nw)
    print('Train set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data)))
    if logFile:
        f = open(logFile,'a')
        f.write('\nIncluding dpil data now\n')
        f.write('Batch_size: '+str(bs)+'\nTrain set: '+str(len(train_set))+' | Validation_set: '+str(len(val_set))+' | Test_set: '+str(len(test_set))+'\n'+'Train_batches: '+str(len(train_data))+' | Validation_batches: '+str(len(val_data))+' | Test_batches: '+str(len(test_data))+'\n')
        f.close()
    # if 'with best model' in hyper_params['include_asha_data']:
    #     checkpoint = torch.load(hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_best.pth.tar')
    #     print(model.load_state_dict(checkpoint['state_dict']))
    for i in range(hyper_params['epochs'],hyper_params['epochs']+2):
        print('-'*50,'Epoch',i+1,'-'*50)
        tloss = model.start_training(i, hyper_params['run_ID'], optim, train_data, val_data, test_data, logging=logFile)
        train_loss.append(tloss)
        vloss, vacc = model.validation(i, hyper_params['run_ID'], val_data, logging=logFile)
        val_loss.append(vloss)
        # torch.save({'state_dict': model.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_epoch_'+str(i)+'.pth.tar')
    model.start_testing(hyper_params['run_ID'],test_data,logFile)

else:
    hyper_params['data_source'] = 'dpil'
    test_set = mydataset(hyper_params['bert_model_name'],'test', hyper_params['max_len'], hyper_params['data_source']) #torch.utils.data.random_split(dataset,[int(len(dataset)*train_ratio), int(len(dataset)*val_ratio), len(dataset)-(int(len(dataset)*train_ratio) + int(len(dataset)*val_ratio))])
    test_data = data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=nw)
    print('Test_set: '+str(len(test_set))+' | Test_batches: '+str(len(test_data)))
    if logFile:
        f = open(logFile,'a')
        f.write('\Testing on dpil data now\n')
        f.write('Batch_size: '+str(bs)+'\n'+'Test_set: '+str(len(test_set))+'\n'+'Test_batches: '+str(len(test_data))+'\n')
        f.close()
    # if 'with best model' in hyper_params['include_asha_data']:
    #     checkpoint = torch.load(hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_best.pth.tar')
    #     print(model.load_state_dict(checkpoint['state_dict']))
    model.start_testing(hyper_params['run_ID'],test_data,logFile,e=hyper_params['epochs']+1, iter=i, message='train')


ttaken = round(time.time() - start_time, 3)
print(ttaken, 'secs')
if logFile:
    f = open(logFile, 'a')
    f.write(str(ttaken)+' secs\n\n')
    f.close()

print(model.predict('मशहूर कोरियोग्राफर सरोज खान नहीं रहीं','सरोज खान नहीं रहीं'))

import matplotlib.pyplot as plt
def plotStats(tL, vL, message, tLlabels="Training Loss", vLlabels="Validation Loss", width=15, xlabels="Epochs", ylim='', save=True):
    fig1, ax = plt.subplots(1,1, figsize=(width,5))
    fig1.suptitle(message)
    plt.subplots_adjust(left=0.04, bottom=0.06, right=0.99, top=0.86, wspace=0.10, hspace=0.53)
    ax.plot(tL, label=tLlabels)
    ax.plot(vL, label=vLlabels)
    ax.set_xlabel(xlabels)
    if(ylim=='zero to one'):
        ax.set_ylim([-0.2,1.2])
    ax.legend()
    ax.grid()
    if save:
        plt.savefig(hyper_params['my_datapath']+'run_ID_'+str(hyper_params['run_ID'])+'.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return

print('Plotting stats...')
# plotStats(train_loss, val_loss, "Training Plots")


# input('Wait')
# SAVE

# torch.save({'state_dict': model.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_final.pth.tar')
# checkpoint = torch.load(hyper_params['my_datapath']+'state_dict/'+str(hyper_params['run_ID'])+'_best.pth.tar')
# print(model.load_state_dict(checkpoint['state_dict']))
# if logFile:
#         f = open(logFile,'a')
#         f.write('\nOn the best model\n')
#         f.close()
# model.start_testing(hyper_params['run_ID'],test_data,logFile)

if logFile:
    f = open(logFile, 'a')
    stamp = datetime.datetime.now(IST).strftime("%c")
    f.write(stamp+'\n')
    f.close()