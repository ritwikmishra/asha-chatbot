# from inltk.inltk import get_sentence_similarity
# from inltk.inltk import get_similar_sentences
import random
import numpy as np
import pandas as pd
import stanza
import pickle 
import string
import re
from tqdm import tqdm
from nltk import Tree
from sentence_transformers import SentenceTransformer, util
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import XLMTokenizer, XLMForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import torch
import time

tqdm.pandas()

def clean(hi_str):
    puncts = string.punctuation+"‟“’❝❞‚‘‛❛❜❟"
    trans_string = hi_str.maketrans("१२३४५६७८९०","1234567890")
    hi_str = hi_str.translate(trans_string)
    hi_str = re.sub(r'[a-z]+','',hi_str.lower())
    hi_str = hi_str.translate(str.maketrans(puncts,' '*len(puncts))).lower().replace('|','').replace('।','')
    hi_str = re.sub(r'\s+',' ',hi_str).strip()
    return hi_str

def lemmatize_keep_nouns(s, nlp):
    s = clean(s)
    doc = nlp(s)
    ml = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos in ['NOUN','PROPN','VERB']:
                ml.append(word.lemma)
    s = ' '.join(ml)
    return s

def wfm(s1,s2,nlp):
    s1 = set(lemmatize_keep_nouns(s1,nlp).split())
    s2 = set(lemmatize_keep_nouns(s2,nlp).split())
    p = (len(s1) - len(s1-s2))/(len(s2)) if len(s2)!=0 else 0
    r = (len(s1) - len(s1-s2))/(len(s1)) if len(s1)!=0 else 0
    f = ((2*p*r)/(p+r)) if p+r != 0 else 0
    return f
    

def wfm_list(l1,l2,nlp): # l1 is list of suggestions at rank 1, l2 is user query. Length of both are same.
    ml = []
    for s1,s2 in zip(l1,l2): # take one by one
        ml.append(wfm(s1,s2,nlp))
    return ml

def add_word_groups(s):
    word_groups = [ 'बेनिफिट्स फायदे'.split(), 'विटामिन्स पोषण'.split(),  'इंजेक्शन टीके'.split(), 'स्वेलिंग सूजन'.split(), ["शुगर","डायबिटीज","मधुमय"], 
    'डेंजर खतरा'.split(), 'आयल तेल'.split(), 'जौंडिस पीलिया'.split(), 'वक्सीनशन टीकाकरण'.split(), 'स्टार्ट शुरू'.split(), 
    'सोप साबुन'.split(), 'हनी शहद'.split(), 'मीनोपॉज रजोनिवृत्ति'.split(), 'अटैचमेंट लगाव'.split(), ["दूध","फीड","ब्रेस्टफीड","स्तनपान"], 
    ["वजन","वज़न","वेट"], ["बुखार","फीवर"], 'पेरासिटामोल पीसीएम'.split(), 'टेम्परेचर तापमान'.split(), ["मासिक","पीरियड्स","माहवारी"], ["छाती","स्तन","ब्रैस्ट"],
    ["ट्विन्स","जुड़वा"], ["नाभि","सुंडी"],["एबॉर्शन","गर्भपात"],["सेक्शन","ऑपरेशन"],["प्रेग्नेंट","ग़र्भवति"],["फॅमिली","परिवार"]]

    s = s.split()
    for i in range(len(s)):
        for g in word_groups:
            if s[i] in g:
                s[i] = ' या '.join(g)
                break
    return ' '.join(s)

dfc = 1
def getTopN(df,q,nlp,model,mymodel,SimModel, show=True):
    # returns a sorted Dataframe, sorted on the basis of similarity scores 
    v = 5.0
    if "Lem" == q.split()[0]: # सीता
        if show:
            print('asking SITA '*5)
        v = 1.0
    elif "DTP" == q.split()[0]: # गंगा
        if show:
            print('asking GANGA '*5)
        v = 1.1
    elif "BERT_spc" == q.split()[0]: #काली
        if show:
            print('asking KAALI '*5)
        v = 3.0
    elif "BERT_cos" == q.split()[0]: #रेनू
        if show:
            print('asking RENU '*5)
        v = 2.0
    elif "BERT_sim" == q.split()[0]:
        if show:
            print('asking Siamese '*5)
        v = 4.0
    elif "E46" == q.split()[0]:
        if show:
            print('Ensemble E46 '*5)
        v = 5.0
    elif "E46.1" == q.split()[0]:
        if show:
            print('Ensemble E46 without cos '*5)
        v = 5.1
    elif "E46.2" == q.split()[0]:
        if show:
            print('Ensemble E46 without dtp '*5)
        v = 5.2
    elif "E46.3" == q.split()[0]:
        if show:
            print('Ensemble E46 without spc '*5)
        v = 5.3
    elif "E46.4" == q.split()[0]:
        if show:
            print('Ensemble E46 without sus '*5)
        v = 5.4
    elif "E46.5" == q.split()[0]:
        if show:
            print('Ensemble E46 without qfm '*5)
        v = 5.5
    else:
        print('Default is v5.0 Ensemble')

    q = ' '.join(q.split()[1:])
    df = keyMatching(df,q,nlp,model,mymodel,SimModel,v)
    return df

def lemmatize(s, nlp):
    s = s.replace(',','').replace('!','').replace('  ',' ')
    doc = nlp(s)
    ml = []
    for sentence in doc.sentences:
        for word in sentence.words:
            ml.append(word.lemma)
    s = ' '.join(ml)
    return s

def retain(ms, s): # so that words in ms (which are randomly arranged bcoz of set) gets aligned with s. As such it is not important and can be removed.
    ms2 = s.split(' ')
    for x in ms2:
        if x not in ms:
            ms2.remove(x)
    return ms2

#            s q
def kMscore(s2,s1,stp,nlp):
    # s2 is from database (predicted)
    # s1 is query from user (gold)

    global dfc

    s2  = s2.translate(str.maketrans('', '', string.punctuation))
    dfc+=1
    r = 0.0
    p = 0.0
    
    # s1 = s1.replace(',','').replace('?','').replace('!','').replace('  ',' ')
    # s2 = s2.replace(',','').replace('?','').replace('!','').replace('  ',' ')
    # print(" ",s1,"\n",s2,"\n")
    # s1 = lemmatize(s1,nlp)
    # s2 = lemmatize(s2,nlp)
    # print(" ",s1,"\n",s2)
    ms1 = set(s1.split(' '))
    ms2 = set(s2.split(' '))
    # print(list(stp)[:20])
    ms1 = ms1 - stp
    ms2 = ms2 - stp
    ms1 = list(ms1)
    ms2 = list(ms2)
    ms1 = retain(ms1,s1)
    ms2 = retain(ms2,s2)
    lms1 = len(ms1)
    lms2 = len(ms2)
    # print(' '.join(ms1))
    # print(ms2)
    c = 0
    for x in ms1:
        if (x in ms2):
            c+=1
            ms2.remove(x)
            
    r = c/lms1
    p = c/lms2
    if p+r == 0:
        f1 = 0
    else:
        f1 = (2*p*r)/(p+r)
    # input('wait')
    return f1

def merge_repeating_suggestions(dff, text_col, score_col):
    # print(dff)
    df2 = pd.DataFrame([],columns=dff.columns)
    count = 0
    for q in list(dff[text_col].unique()):
        tdf = dff[dff[text_col] == q].reset_index(drop=True)
        for c in tdf.columns:
            df2.at[count,c] = tdf.loc[0,c]
        # print('-'*100)
        # print(tdf[score_col], q)
        df2.at[count,score_col] = sum(list(tdf[score_col]))
        count+=1
    return df2

def keyMatching(df,q, nlp, embedder, mymodel, SimModel, version=1.0):
    global dfc
    dfc = 1

    f = open('hi_stopwords.pkl','rb')
    stp = pickle.load(f)
    f.close()

    stp = set(stp)

    if version == 1.0:
        q = lemmatize(q,nlp)
        df['score'] = df['lemmaQuestion'].apply(kMscore,args=[q,stp,nlp])
    elif version == 1.1:
        q = concise(q,nlp)
        df['score'] = df['c_lemmaQuestion'].apply(kMscore,args=[q,stp,nlp])
    elif version == 2.0:
        sList = list(df['Question'])
        # corpus_embeddings = embedder.encode(sList, convert_to_tensor=True)
        corpus_embeddings = torch.tensor(np.array(df['embeddings'].tolist()))
        query_embedding = embedder.encode(q, convert_to_tensor=True).to('cpu')
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.to('cpu')
        df['score'] = cos_scores
    elif version == 3.0:
        df['score'] = mymodel.databaseSearch(q,df) #df['Question'].progress_apply(lambda d: 100*mymodel.predict(q,d,tokenizer))
    elif version == 4.0:
        sList = list(df['Question_clean'])
        df['score'] = SimModel.batch_predict(q, sList, 41, tokenizer)

    elif version == 5.0:
        # print('\n\t\t -------- running on DTP --------')
        q1 = concise(add_word_groups(q),nlp)
        df1 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df1['score'] = list(df['c_lemmaQuestion'].apply(kMscore,args=[q1,stp,nlp]))
        df1 = df1.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]
        print('DTP',df1)

        # print('\n\t\t -------- running on BERT_spc --------')
        df2 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df2['score'] = list(mymodel.databaseSearch(q,df))
        df2 = df2.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]
        print('BERT_spc',df2)

        # print('\n\t\t -------- running on BERT_cos --------')
        df3 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        sList = list(df['Question'])
        # corpus_embeddings = embedder.encode(sList, convert_to_tensor=True)
        corpus_embeddings = torch.tensor(np.array(df['embeddings'].tolist()))
        query_embedding = embedder.encode(add_word_groups(q), convert_to_tensor=True).to('cpu')
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.to('cpu')
        df3['score'] = list(cos_scores)
        df3 = df3.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]
        print('BERT_cos',df3)

        dff = pd.concat([df1,df2,df3]).reset_index(drop=True)
        dff['score'] = dff['score'].astype(float)

        df = merge_repeating_suggestions(dff, text_col='Question', score_col='score')
        print('Ensemble', df)
    
    elif version == 5.1:
        # print('\n\t\t -------- running on DTP --------')
        q1 = concise(add_word_groups(q),nlp)
        df1 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df1['score'] = list(df['c_lemmaQuestion'].apply(kMscore,args=[q1,stp,nlp]))
        df1 = df1.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        # print('\n\t\t -------- running on BERT_spc --------')
        df2 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df2['score'] = list(mymodel.databaseSearch(q,df))
        df2 = df2.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        dff = pd.concat([df1,df2]).reset_index(drop=True)
        dff['score'] = dff['score'].astype(float)

        df = merge_repeating_suggestions(dff, text_col='Question', score_col='score')
    
    elif version == 5.2:
        # print('\n\t\t -------- running on BERT_spc --------')
        df2 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df2['score'] = list(mymodel.databaseSearch(q,df))
        df2 = df2.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        # print('\n\t\t -------- running on BERT_cos --------')
        df3 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        sList = list(df['Question'])
        # corpus_embeddings = embedder.encode(sList, convert_to_tensor=True)
        corpus_embeddings = torch.tensor(np.array(df['embeddings'].tolist()))
        query_embedding = embedder.encode(add_word_groups(q), convert_to_tensor=True).to('cpu')
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.to('cpu')
        df3['score'] = list(cos_scores)
        df3 = df3.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        dff = pd.concat([df2,df3]).reset_index(drop=True)
        dff['score'] = dff['score'].astype(float)

        df = merge_repeating_suggestions(dff, text_col='Question', score_col='score')

    elif version == 5.3:
        # print('\n\t\t -------- running on DTP --------')
        q1 = concise(add_word_groups(q),nlp)
        df1 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        df1['score'] = list(df['c_lemmaQuestion'].apply(kMscore,args=[q1,stp,nlp]))
        df1 = df1.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        # print('\n\t\t -------- running on BERT_cos --------')
        df3 = df[['file_id', 'topic', 'category', 'Question', 'Question_clean', 'qStart', 'qEnd', 'Answer', 'aStart', 'aEnd']].copy()
        sList = list(df['Question'])
        # corpus_embeddings = embedder.encode(sList, convert_to_tensor=True)
        corpus_embeddings = torch.tensor(np.array(df['embeddings'].tolist()))
        query_embedding = embedder.encode(add_word_groups(q), convert_to_tensor=True).to('cpu')
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.to('cpu')
        df3['score'] = list(cos_scores)
        df3 = df3.sort_values(['score'], ascending=[False]).reset_index(drop=True)[:3]

        dff = pd.concat([df1,df3]).reset_index(drop=True)
        dff['score'] = dff['score'].astype(float)

        df = merge_repeating_suggestions(dff, text_col='Question', score_col='score')


    df = df[['Question','Answer','score','category']]
    df = df.sort_values(['score'], ascending=[False]).reset_index(drop=True)
    #df.to_csv("test.csv",index=False)
    return df


def getTranscriptionsQnA(df):
    a = df
    fids = np.sort(a['file_id'].unique())
    newdf = pd.DataFrame([],columns=df.columns)
    for f in fids:
        c = a[a['file_id']==f]
        c = c.sort_values('start_time')
        c = c[ (c['is_deleted']==0) & (c['relevance']=='Relevant') ]
        newdf = pd.concat([newdf, c])

    newdf.drop_duplicates(subset=['file_id', 'transcription', 'start_time', 'end_time'], keep='first', inplace=True)
    newdf.to_csv('/home/ritwik/git/AshaQnA/transcriptionsQnA.csv', index=None)
    return newdf

def f():
    df = pd.read_csv('allQnA.csv')
    df = df[df['Answer'].str.contains('हमारा अगला')]
    df.to_csv('temp.csv',index=None)
    return 


# from original transcription file (one time process)
# change 75459, 80151, 50015, 41025-41030, 43915-43917, 90126, 44330-44332 Moderator to Doctor
# change 44583, 71508, Irrelevant to Relevant
# change 89736, 89735, 39068, 40972, 40973, 40974, 40975, 40977, 40978,    68337, 62058, 62059, 62063, 62060, 62061, 62064, 62065, 62066  Relevant to Irrelevant
# remove index = 89143, insert आज हमारे साथ डॉ विजय आपकी पिछली ट्रेनिंग नवजात शिशु का तापमान
    # why? bad annotation
# change 72220, 92127, 45276, 74627, 65459, 65460, 61584, 45204, 45360, 45359,
# 45392, 60714, 67420, 67327, 93454, 93480, 43968, 43969, 43970, 86024, 86025,
# 86026, 86064, 89838, 84870, 72049, 72136, 56875, 90620, 90631, 93881, 66450,
# 87637, 77508, 77518, 75472, 75545, 44803, 44841, 46090, 55385, 71534, 71568,
# 82890, 85444, 85445, 85475, 82498  Doctor to Moderator
    # why? doctor asking question
# change 53796 Irrelevant to Relevant
    # why? doctor being rude, 

def convertToQnA(df):
    ml = []
    q = ""
    a = ""
    fid = topic = qS = qE = aS = aE = 0
    sawAnswer = False
    noise = [ #ordering is important here
        'हमारा अगला सवाल है ',
        'हमारा अगला सवाल है, ',
        'हमारा अगला सवाल है',
        'हमारा अगला सवाल ',
        'हमारा आज का पहला सवाल है, ',
        'हमारा आज का पहला सवाल है ',
        '<सर> का आज का हमारा पहला सवाल है ',
        '<सर> आज का हमारा पहला सवाल है ',
        'हमारा पहला सवाल है ',
        'हमारा आज का अगला सवाल है ',
        'अगला सवाल है ',
        'हमारा आज का आखरी सवाल है ',
        'हमारा पहला सवालों है ',
        ' मैंने इस सवाल का जवाब दे दिया है',
        'आज का हमारा सवाल है ',
        'हमारा आज का अगला प्रशन है ',
        'हमारा अगला परशान ',
        'हमारा अगला प्रश्न है की ',
        'हमारा अगला प्रश्न है की',
        'हमारा अगला प्रशन है, ',
        'हमारा अगला प्रशन है,',
        'हमारा अगला प्रशन है ',
        'हमारा अगला प्र्शन है ',
        'हमारा अगला प्रश्न है ',
        'हमारा अगला प्रशन है',
        'हमारा अगला प्रश्न है',
        'हमारा अगला प्रशन ',
        'हमारा अगला प्र्शन है',
        'हमारा अगला <कुएस्शन> है, ',
        'हमारा अगला <कुएस्शन> है ',
        'हमारा अगला <कुएस्शन> है',
        'हमारा अगला प्र्शन प्र्शन है ',
        'हमारा अगलाप्रशन है ',
        'हमारा अगला राशन है ',
        'आज का हमारा पहला प्रशन होगा ',
        'आज का पहला प्रश्न है ',
        'आज का हमारा सबसे पहला प्रश्न है ',
        'आज का हमारा पहला प्रशन है, ',
        'आज का हमारा पहला प्रशन है ',
        'आज का हमारा पहला प्रशन ',
        'हमारा पहला प्रशन है ',
        'हमारा पहला प्रश्न है ',
        'हमारा पहला प्रशन ',
        'हमारा प्रश्न है ',
        'हमाराअगला प्रशन है ',
        'हमारा अगला प्रशनहै ',
        'पहला प्रशन है ',
        'अगला प्रश्न है ',
        'हमारा अगला '
    ] #प्रशन is different than प्रश्न, you cannot see it in linux, whatsapp this to yourself
    for i in range(len(df)):
        if (df['speaker'].iloc[i]=="Moderator"):
            if (sawAnswer):
                q = q.replace('\n',' ')
                q = re.sub(r' +',' ',q).strip()
                for n in noise:
                    q = q.replace(n,'')
                a = a.replace('\n',' ')
                a = re.sub(r' +',' ',a).strip()
                ml.append([fid,topic,q,qS,qE,a,aS,aE])
                q = a = ""
                sawAnswer = False
            if (q==""): # <-------- new question seen
                fid = df['file_id'].iloc[i]
                topic = df['topic'].iloc[i]
                qS = df['start_time'].iloc[i]
            q = q + ' ' + df['transcription'].iloc[i]
            qE = df['end_time'].iloc[i]
        else: # <------ Doctor encountered
            sawAnswer = True
            if (a==""):
                aS = df['start_time'].iloc[i]
            a = a + ' ' + df['transcription'].iloc[i]
            aE = df['end_time'].iloc[i]

    newdf = pd.DataFrame(ml,columns=['file_id','topic','Question','qStart','qEnd','Answer','aStart','aEnd'])
    newdf = newdf[newdf['Question']!='(sil)']
    newdf = newdf[~newdf['Question'].str.contains('<सर>')]
    
    mdf = pd.read_csv('mother-questions.csv')
    newdf = pd.concat([newdf, mdf])
    newdf.to_csv('allQnA.csv', index=None) # 23 bogus rows
    return newdf

class dmatrix():
    def __init__(self,ml):
        '''
        if cell ij has 1, then it indicates that word j is a child of word i. Hence parents are stored in rows, and children are stored in columns.
        '''
        self.words = ml
        self.mat = np.zeros((len(ml),len(ml)))
        for d in ml:
            if d.head != 0:
                self.mat[d.head-1, int(d.id)-1] = 1
        self.root_value = self.mat.sum(axis=0).argmin()

    def show(self):
        print("    ", end='')
        for i in range(len(self.words)):
            print((i+1)%10, end='    ')
        print()
        for i in range(len(self.words)):
            print((i+1)%10, list(self.mat[i]))

    def get_root(self):
        return self.root_value

    def set_root(self, n):
        self.root_value = n
        return


    def children(self, n):
        c_indexList = np.where(self.mat[n] == 1)[0]
        children = []
        for cindx in c_indexList:
            children.append(self.words[int(cindx)])
        return children

    def parent(self, n):
        if self.root_value == n:
            return -1
        else:
            return self.words[np.where(self.mat[:,n] == 1)[0][0]]

    def text(self, n):
        return self.words[n].text

    def node_text(self, n):
        w = self.words[n]
        s = '^'+str(w.id)+'_'+w.text+'&'+w.xpos+'@'+w.upos+'%'+w.deprel+'$'
        s = w.text+'_'+w.deprel
        s = '  '+str(w.id)+'_'+w.text+'  '+'\n'+w.deprel+'\n'+w.upos
        return s

    def n_ancestors(self, n):
        '''
        this calculates number of nodes which has this given node as an ancestor
        or in simple language
        number of possible children/grand children/great grand children etc of this node
        '''
        a = len(self.children(n))
        child_queue = [int(child.id)-1 for child in self.children(n)]
        while len(child_queue) > 0:
            tl = [ int(child.id)-1 for child in self.children(child_queue[0])]
            a = a + len(tl)
            child_queue = child_queue + tl 
            child_queue.reverse()
            child_queue.pop()
            child_queue.reverse()
        return a

class myBert(torch.nn.Module):
    def __init__(self,d,hyper_params,which_criterion):
        super(myBert, self).__init__()
        self.bert_model_name = hyper_params['bert_model_name']
        self.label_smoothing = hyper_params['label_smoothing']
        self.max_len = hyper_params['max_len']
        self.arch = hyper_params['architecture']
        self.hyper_params = hyper_params

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

            allQnA['score'] = self.databaseSearch(q,allQnA)

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
        md = calc_metrics(df)

        file = open(hyper_params['my_datapath']+'train_hold_out_metrics_'+str(self.hyper_params['run_ID'])+'.txt','a')
        file.write('Epoch_'+str(epoch)+' Iter='+str(iter)+' ')
        for k in md.keys():
            file.write(k+':'+str(md[k])+' ')
        file.write('\n')
        file.close()

        if md['avg_sr'] > self.best_sr:
            self.best_sr = md['avg_sr']
            df.to_csv(hyper_params['my_datapath']+'run_ID_'+str(run_ID)+'_best_sr.csv',index=None)
            torch.save({'state_dict': self.state_dict()},  hyper_params['my_datapath']+'state_dict/'+str(run_ID)+'_best_sr.pth.tar')

        return

    def start_training(self, e, run_ID, optim, train_data, val_data, test_data, logging=''):
        ttime = time.time()
        self.model.train()
        ep_train_loss = []

        for i, (input_ids, token_type_ids, attention_mask, y) in tqdm(enumerate(train_data), desc='training /'+str(len(train_data)),total=len(train_data), ncols=100):
            self.model.train()
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
        self.model.eval()
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
                f.write('Accuracy '+str(round(100*int(sum(y_true == y_pred))/len(y_true),3))+'\n')
                f.close()
        return ep_valid_loss, ep_valid_acc

    def start_testing(self, run_ID, test_data, logging='', e = 0, iter=0, message=''):
        ttime = time.time()
        self.model.eval()
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


        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)

        if message == 'train':
            file = open(hyper_params['my_datapath']+'train_test_acc_'+str(self.hyper_params['run_ID'])+'.txt','a')
            file.write('Epoch_'+str(e)+' Iter='+str(iter)+' '+str(100*int(sum(y_true == y_pred))/len(y_true))+'\n')
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
            print('\nAccuracy',100*int(sum(y_true == y_pred))/len(y_true))
            ts+=('\nAccuracy '+str(round(100*int(sum(y_true == y_pred))/len(y_true),4))+'\n')
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
    def databaseSearch(self, q, df):
        self.model.eval()
        tokenizer=self.mytokenizer_for_hold_out
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

class SiameseNetwork(nn.Module):
    def __init__(self,emb_opt,res_opt,bert_model_name,device,sig):
        super(SiameseNetwork, self).__init__()
        self.emb_opt = emb_opt
        self.res_opt = res_opt
        self.bert_model_name = bert_model_name
        self.device = device
        self.sig = sig
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        var_fc_in_dimension = 768*2 if res_opt == 3 else 768 
        self.model = AutoModel.from_pretrained(bert_model_name)
        self.fc = nn.Sequential(
                      nn.Linear(var_fc_in_dimension, 90), 
                      nn.ReLU(inplace=True),
                      nn.Linear(90, 1))
        
    def make_sentence_embeddings(self,output):
      which_way = self.emb_opt
      result = []
      if which_way == 1 :
        # CLS
        result = output['last_hidden_state'][:,0,:]
        #print(result.shape)
        result = result.view(-1,768) # [batch_size,768]
      elif which_way == 2 :
        result = torch.mean(output['last_hidden_state'], dim=1) # tensor = [batch_size, 768]
        #print("Mean : ", result.shape)
      elif which_way == 3 :
          # pooler
          result = output['pooler_output']
      elif which_way == 4:
          # starting 2 avg
          result = output['hidden_states']
          #print("type result : ",type(result))
          #print("HSSS shape : ", len(result))
          #print("HSSS 0th shape : ", result[0].shape) # [batch_size, 41, 768]
          #print("HSSS 0th shape : ", result[0][:,0].shape) # tensor = [batch_size, 768]
          #input()
          
          tensor_f = torch.stack([ result[0][:,0], result[1][:,0] ])
          #print("tensor shape : ", tensor_f.shape ) # tensor = [2, batch_size, 768]
          result = torch.mean(tensor_f, dim = 0)
          #print("Result shape : ", result.shape ) # tensor = [batch_size, 768]
          #input()

      elif which_way == 5:
          #  last 2 layer avg
          result = output['hidden_states']
          
          tensor_f = torch.stack([ result[-2][:,0], result[-3][:,0] ])

          result = torch.mean(tensor_f, dim = 0)

      return result

    def mix_result(self,output1,output2):
      which_way = self.res_opt
      result = []
      if which_way == 1 :
        result = torch.abs(output1 - output2)
        
        result = self.fc(result)
      elif which_way == 2 :
        result = torch.abs(output1 + output2)

        result = self.fc(result)
      elif which_way == 3 :
        result = torch.stack([output1, output2], dim=1) # tensor = [batch_size, 2, 768]
        # print("After Stacking : ",result.shape)
        result = result.view(-1,768*2) # tensor = [batch_size, 1536]
        # print("After Stacking and reshape : ",result.shape) 
        result = self.fc(result)
      elif which_way == 4:
          result = self.cos(output1, output2)
          result = torch.unsqueeze(result, 1)
          
      return result

    def forward_once(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids, token_type_ids, attention_mask,output_hidden_states=True) # dict = ['last_hidden_state', 'pooler_output']
        # output['last_hidden_state']  tensor = [batch_size, max_length, 768]
        # output['pooler_output']  tensor = [batch_size, 768]

        #print(output.keys())
        #rint("Before Sequeeze pooler : ", output['pooler_output'].shape)

        #output = output['last_hidden_state'] # tensor = [batch_size, max_length, 768]

        #print("Before Sequeeze : ",output.shape)
        #if output.shape[0] != 1: 
        #    output = output.squeeze(0) # tensor = [batch_size, max_length, 768]

        #print("After Sequeeze",output.shape)
        output = self.make_sentence_embeddings(output)  # tensor = [batch_size, 768]

        return output

    def forward(self, input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2):
        #print("SSSS : ", input_ids1.shape)
        output1 = self.forward_once(input_ids1, token_type_ids1, attention_mask1)  # tensor = [batch_size, 768]
        output2 = self.forward_once(input_ids2, token_type_ids2, attention_mask2)  # tensor = [batch_size, 768]
        #print("Output unsqzz option",output1.shape, output2.shape)
        #print("Output mix before",output1.shape, output2.shape)

        output = self.mix_result(output1,output2) # tensor = [batch_size, 1536]

        #print("Output mix : ",output.shape)

        #output = self.fc(output) # tensor = [batch_size, 1]
        #print("FC : ",output.shape)

        if self.sig ==1:
            fin_out = torch.sigmoid(output) # tensor = [batch_size, 1]
        else:
            # print("SSS",output.shape)
            fin_out = output
        #print("Sigmoid : ",sig_out.shape)
        return fin_out
    
    def batch_predict(self, sent_target, sent_batch, max_len, tokenizer):

        class dataset_vec(data.Dataset):
            def __init__(self,bert_model_name,tokenizer,seed=123):

                # tokenizer = AutoTokenizer.from_pretrained(get_model_name(model_id))
                enc1 = tokenizer(sent_target, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
                
                input_ids1 = enc1['input_ids'].numpy()
                token_type_ids1 = enc1['token_type_ids'].numpy()
                attention_mask1 = enc1['attention_mask'].numpy()

                self.sent_len = len(sent_batch)

                input_ids1_list = [input_ids1] * self.sent_len
                token_type_ids1_list = [token_type_ids1]*self.sent_len
                attention_mask1_list = [attention_mask1]*self.sent_len

                self.input_ids1_vec = torch.tensor(input_ids1_list).squeeze(1)
                self.token_type_ids1_vec = torch.tensor(token_type_ids1_list).squeeze(1)
                self.attention_mask1_vec = torch.tensor(attention_mask1_list).squeeze(1)

                input_ids2_list = []
                token_type_ids2_list = []
                attention_mask2_list = []
                for sent in sent_batch:
                    enc2 = tokenizer(sent, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
                    
                    input_ids2 = enc2['input_ids'].numpy()
                    token_type_ids2 = enc2['token_type_ids'].numpy()
                    attention_mask2 = enc2['attention_mask'].numpy()

                    input_ids2_list.append(input_ids2)
                    token_type_ids2_list.append(token_type_ids2)
                    attention_mask2_list.append(attention_mask2)

                
                self.input_ids2_vec = torch.tensor(input_ids2_list).squeeze(1)
                self.token_type_ids2_vec = torch.tensor(token_type_ids2_list).squeeze(1)
                self.attention_mask2_vec = torch.tensor(attention_mask2_list).squeeze(1)
                

                del enc1, enc2, input_ids1_list, token_type_ids1_list, attention_mask1_list, input_ids2_list, token_type_ids2_list, attention_mask2_list

            def __getitem__(self,index):
                return self.input_ids1_vec[index], self.token_type_ids1_vec[index], self.attention_mask1_vec[index], self.input_ids2_vec[index], self.token_type_ids2_vec[index], self.attention_mask2_vec[index]
            
            def __len__(self):
                return self.sent_len

        final = []
        device = self.device

        dataset = dataset_vec(self.bert_model_name,tokenizer)
        batch_data = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # print("\n Total Batches : ",len(batch_data))

        for i, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2) in enumerate(batch_data):
            
            input_ids1 = input_ids1.to(device)
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)

            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)

            with torch.no_grad():
                out = self.forward(input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2).squeeze(1)
             
            out = out.to("cpu") 
            # out[out > thresh] = 1.0
            # out[out < thresh] = 0.0

            if out.shape[0] != 1:
                final += out.squeeze().tolist()
            else:
                final += out.tolist()

            del input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, out

        return final



def to_nltk_tree(mat, n):
    children = mat.children(n)
    if len(children) != 0:
        return Tree(mat.node_text(n), [to_nltk_tree(mat, int(child.id)-1) for child in children])
    else:
        return mat.node_text(n)

def prune_advice(mat):
    root_children = [child.text for child in mat.children(mat.get_root()) ] # children of the root
    if (
    "सलाह" in root_children or 
    "इलाज" in root_children or 
    ("चाहिए" in root_children and mat.words[mat.get_root()].lemma == "कर") or 
    ("क्या" in root_children and mat.words[mat.get_root()].lemma == "कर")
    ):
        r_c_ids = [int(child.id) for child in mat.children(mat.get_root()) ] # ids of root children
        mx_anc = r_c_ids[0]-1
        for i in r_c_ids:
            if mat.n_ancestors(i-1) > mat.n_ancestors(mx_anc): # i-1 is done for 0 indexing shift
                mx_anc = i-1
        mat.set_root(mx_anc)
    return

def check(check_rel):
    '''
    creates the dmatrix and shows (one by one) the depedency tree of
    those Questions (from our database) which has at least one instance of the given dependency relation "check_rel"
    '''
    df = pd.read_csv('allQnA.csv')
    ql = list(df['Question'])
    print('='*200)
    for q in ql:
        q = q.translate(str.maketrans('', '', string.punctuation))
        doc = nlp(q)
        ml = []
        for sent in doc.sentences:
            for word in sent.words:
                ml.append(word)
        flag = False
        for m in ml:
            if m.deprel == check_rel:
                flag = True
                break
        if flag:
            m = dmatrix(ml)
            # prune_salah(m)
            tree = to_nltk_tree(m, m.get_root())
            print(q)
            tree.pretty_print()
            print('-'*200)
            input('wait')
    return

def remove_relations(m):
    ignore_dep_rel = ['dep', 'displocated', 'discourse', 'expl', 'cc','case','aux','aux:pass','mark']

    i = 0
    while  i<len(m.words):
        if m.words[i].deprel in ignore_dep_rel:
            m.mat[:,i] = 0
        i+=1

    return 


def make_child_a_parent(m, child_id, parent_id):
    m.mat[parent_id,child_id] = 0
    m.mat[child_id] = m.mat[child_id] + m.mat[parent_id]
    m.mat[parent_id] = 0 
    if (m.get_root() == parent_id):
        m.set_root(child_id)
    elif (m.words[parent_id].deprel == 'root'):
        pass
    else:
        grand_parent_id = int(m.parent(parent_id).id)-1
        m.mat[grand_parent_id, parent_id] = 0
        m.mat[grand_parent_id, child_id] = 1
    m.words[child_id].deprel = m.words[parent_id].deprel
    return 

def compound_expansion(m):
    for i in range(len(m.words)):
        w = m.words[i]
        if w.deprel == "compound" and m.parent(i)!=-1 and m.parent(i).upos == "VERB":
            make_child_a_parent(m,i,int(m.parent(i).id)-1)

    return

def post_order(m, n, s):
    children = m.children(n)
    if len(children) == 0:
        return m.words[n].text, s+m.words[n].lemma+' '
    children_ids = [int(child.id)-1 for child in children]
    for i in children_ids:
        a, s = post_order(m,i,s)
        # if a:
        #     print(a, end=' ')
    # print(m.words[n].text, end=' ')
    s = s+m.words[n].lemma+' '
    return '', s

def concise(text, nlp, showPlease=False):
    doc = nlp(text)
    # print(doc)
    s = ""
    for sent in doc.sentences:
        ml = []
        for word in sent.words:
            ml.append(word)
    
        m = dmatrix(ml)
        
        if (showPlease):
            print('Beginning')
            print(sent.text)
            m.show()
            tree = to_nltk_tree(m, m.get_root())
            tree.pretty_print()

        prune_advice(m)

        if (showPlease):
            print('\nAfter pruning advice')
            tree = to_nltk_tree(m, m.get_root())
            tree.pretty_print()

        remove_relations(m)

        if (showPlease):
            print('\nAfter removing relations') #see definition chota sa function hai
            tree = to_nltk_tree(m, m.get_root())
            tree.pretty_print()

        compound_expansion(m)

        if (showPlease):
            print('\nAfter compund expansion')
            tree = to_nltk_tree(m, m.get_root())
            tree.pretty_print()

        _, s2 = post_order(m, m.get_root(),'')

        if (showPlease):
            print('\nAfter post order traversal')
            print(s2)

        s = s + s2 + ' '
    
    if(showPlease):
        print('\nFinal output')

    return s



if __name__ == '__main__':

    # ------------- Get allQnA,csv from raw transcriptions.csv ----
    # df = pd.read_csv('transcriptions.csv')
    # df = getTranscriptionsQnA(df)
    # df = df.dropna(subset=['transcription'])
    # df = convertToQnA(df)
    # exit()
    # ---------------------------------------------------------------


    # ------------ Generate lemmas of the question ------------------
    # stanza.download('hi')
    # nlp = stanza.Pipeline('hi')

    # df = pd.read_csv('allQnA.csv')
    # tqdm.pandas()
    # df['lemmaQuestion'] = df['Question'].progress_apply(lambda d: clean(d))
    # df['lemmaQuestion'] = df['lemmaQuestion'].progress_apply(lemmatize, args=[nlp])
    # df.to_csv('allQnA.csv', index=None)
    # exit()
    # ---------------------------------------------------------------


    # --------------- Generate concise lemmas -----------------------
    # nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos,lemma,depparse', verbose=False)
    # df = pd.read_csv('allQnA.csv')
    # tqdm.pandas()
    # df['c_lemmaQuestion'] = df['Question'].progress_apply(lambda d: clean(d))
    # df['c_lemmaQuestion'] = df['c_lemmaQuestion'].progress_apply(concise, args=[nlp])
    # df.to_csv('allQnA.csv', index=None)
    # exit()
    # ---------------------------------------------------------------

    # ----------------- Create Embeddings ---------------------------
    # df = pd.read_csv('allQnA.csv')
    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # tqdm.pandas()
    # df['embeddings'] = ''
    # df['embeddings'] = torch.tensor(model.encode(df['Question'].progress_apply(lambda d: clean(d)).tolist()))
    # torch.save(df,'../ashaData/allQnA.bin')
    # exit()
    # ---------------------------------------------------------------


    # -------------
    # from nltk.corpus import wordnet
    # from nltk.wsd import lesk
    # sent = 'I like apples'
    # ambiguous = 'like'
    # s1 = wordnet.synset("like.v.05")

    # sent = 'I love apples'
    # ambiguous = 'love'
    # s2 = lesk(sent, ambiguous)

    # print(s1.definition(),'###',s2.definition(),s1.wup_similarity(s2))
    # exit()

    # syn = wordnet.synset("like.n.01")
    # s2 = wordnet.synset("love.n.01")
    # print(syn.wup_similarity(s2))

    # # print(syn[0].hypernyms())
    # exit()


    text = 'अगर छोटे बच्चे को गैस बहुत ज़्यादा बनती है तो हम उन्हें क्या सलाह दें'
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = 'चार्ली ब्राउन कौन है'
    # text = 'What fruit do you like?'
    nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos,lemma,depparse', verbose=False)
    print(text)
    print(concise(text, nlp, True))
    exit()

    df = pd.read_csv('allQnA.csv')
    df['c_lemmaQuestion'] = df['Question'].apply(lambda d: d.translate(str.maketrans('', '', string.punctuation)))
    ml = list(df['c_lemmaQuestion'])
    c = 0
    for k,text in enumerate(ml):
        try:
            s = concise(text, nlp)
        except:
            print(text)
            input('wait')
        print(k,'/',len(ml))
    print(c)
    # check('compound')
    exit()
    # ---------------------------------------------------------------

    # print(get_similar_sentences('माँ के निप्पल बहुत अंदर घुसे हुए है और बच्चा निप्पल पकड़ नहीं पा रहा तो क्या सलाह दे?', 10, 'hi'))
    # exit()

    stanza.download('hi')
    # # doc = nlp('बच्चे की डिलीवरी होने के बाद माँ का दूध काम उतर रहा है दूध ज़्यादा उतरे इसके लिए क्या करें?')
    # # for sentence in doc.sentences:
    # #     for word in sentence.words:
    # #         print(word.lemma)

    # a = "अगर माँ के निप्पल बहुत अंदर धसे हुए हो और बच्चे को फीड ना करवा पाए तो क्या करे?"
    # b = "माँ के निप्पल बहुत अंदर घुसे हुए है और बच्चा निप्पल पकड़ नहीं पा रहा तो क्या सलाह दे?" #55

    # stp = []
    # for x in range(1,5):
    #     file = open('stopwords'+str(x)+'.txt')
    #     content = file.readlines()[1:]
    #     stp += content
    # stp = set([x.strip('\n') for x in list(set(stp))])
    # print(kMscore(a,b,stp,nlp))

    # exit()

    import pandas as pd
    df = pd.read_excel('10 audio transcription data2.xlsx')
    df['lemmaQuestion'] = df['Question'].apply(lemmatize, args=[nlp])
    df.to_excel('10 audio transcription data2.xlsx',index=None)
    exit()
    # df = df.dropna()
    
    # df = keyMatching(df,'डिलीवरी  शुरू  में  माँ  का  दूध  कम  उतरता  है  तो  दूध  दे?')
    # df.to_csv('temp.csv',index=None)


    a = "अगर माँ के निप्पल बहुत अंदर धसे हुए हो और बच्चे को फीड ना करवा पाए तो क्या करे?"
    b = "माँ के निप्पल बहुत अंदर घुसे हुए है और बच्चा निप्पल पकड़ नहीं पा रहा तो क्या सलाह दे?" #55

    a = "माँ की छाती में दूध पिलाने के बाद सूजन आती है तो क्या सलाह दे?"
    b = "अगर माँ की छाती बच्चे को दूध पिलाने से सूज जाती हो तो क्या करे?" #104

    a = "बच्चे की डिलीवरी होने के बाद माँ का दूध काम उतर रहा है दूध ज़्यादा उतरे इसके लिए क्या करें?"
    b = "डिलीवरी के बाद शुरू में माँ का दूध कम उतरता है तो दूध ज्यादा उतरने के लिए क्या सलाह दे?" #1
    print(get_sentence_similarity(a,b,'hi'))

    # exit()
    stp = []
    for x in range(1,5):
        file = open('stopwords'+str(x)+'.txt')
        content = file.readlines()[1:]
        stp += content
    stp = set([x.strip('\n') for x in list(set(stp))])
    print(kMscore(a,b,stp))


    df=getTopN(df,a)
    df.to_csv('temp.csv',index=None)

"""
 दूध पिलाने से सूज जाती हो तो क्या करे 
 दूध पिलाने से सूजन जाती हो तो क्या करे 

 दूध पिला से सूज जा हो तो क्या कर 
 दूध पिला से सूजन जा हो तो क्या कर

 """

 # माँ के निप्पल बहुत अंदर घुसे हुए है और बच्चा निप्पल पकड़ नहीं पा रहा तो क्या सलाह दे 
 # अगर माँ के निप्पल बहुत अंदर धसे हुए हो और बच्चे को फीड ना करवा पाए तो क्या करे 

 # माँ का निप्पल बहुत अंदर घुस हो है और बच्चा निप्पल पक नहीं पा रह तो क्या सलाह दे 
 # अगर माँ का निप्पल बहुत अंदर धस हो हो और बच्चा को फीड ना करवा पा तो क्या कर
