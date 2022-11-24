'''
this code generates a dataset for Hindi paraphrase
	from inshorts folder

this code generates carefully curated negative sentences based on fscores
'''


from google.colab import drive
drive.mount('/content/gdrive')

# for .tar.xz file
# !tar -xJf "gdrive/My Drive/SentSimHi/data/inshorts-dataset-hi.tar.xz" -C "gdrive/My Drive/SentSimHi/data/"
# for zip file
# !unzip gdrive/My\ Drive/SentSimHi/data/inshorts-dataset-hi.zip -d gdrive/My\ Drive/SentSimHi/data/

# number of files
# !ls gdrive/MyDrive/SentSimHi/data/inshorts-dataset-hi/* | wc -l


def lemmatize(s, nlp):
    s = s.replace(',','').replace('!','').replace('  ',' ')
    doc = nlp(s)
    ml = []
    for sentence in doc.sentences:
        for word in sentence.words:
            ml.append(word.lemma)
    s = ' '.join(ml)
    return s

def get_nouns(s, nlp):
    doc = nlp(s)
    nl = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ['PROPN','NOUN']:
                nl.append(word.text)
    return ' '.join(nl)

def find_negative(s, sl):
    # sl = sl[:99]
    og = sl.copy()
    sl = [val for val in sl if val!=s]
    recall = []
    precision = []
    fscore = []
    cl = []
    for i in range(len(sl)):
        p = sl[i].split()
        t = s.split()
        common = set(p).intersection(set(t))
        recall.append(len(common)/len(t) if len(t)!=0 else 0)
        precision.append(len(common)/len(p) if len(p)!=0 else 0)
        f = 2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]) if (precision[-1]+recall[-1]!=0) else 0
        fscore.append(f)
        cl.append(common)
    # print(set(s.split()))
    # print('\n',sl[np.argmax(recall)],'\n',cl[np.argmax(recall)],'\n', sl[np.argmax(precision)],'\n',cl[np.argmax(precision)],'\n')
    # print(og[og.index(s)])
    return og.index(sl[np.argmax(recall)]), og.index(sl[np.argmax(precision)]), og.index(sl[np.argmax(fscore)])

# CREATING DATASET (new)

!pip install --upgrade tables > temp.txt
!pip install tqdm > temp.txt
!pip install stanza > temp.txt

import stanza
stanza.download('hi')
# from IPython.display import display, HTML
from tqdm import tqdm 
import pickle
import glob
import pandas as pd
import numpy as np
import string
import os

nlp = stanza.Pipeline('hi', use_gpu=True)

with open('gdrive/My Drive/SentSimHi/data/hi_stopwords.pkl', 'rb') as f:
    hi_stopwords = pickle.load(f)

ml = glob.glob("gdrive/My Drive/SentSimHi/data/inshorts-dataset-hi/*.txt")
df = pd.DataFrame([], columns=['fID','Sent', 'Positive'])

print('NUMBER OF FILES',len(ml))
# df = pd.read_hdf('gdrive/MyDrive/SentSimHi/data/df.h5', 'ids')
print('\n\n\n')

j = 0
for m in tqdm(ml, desc='Creating dataframe',position=0, leave=True):
    f = open(m,'r')
    df.at[j,'fID'] = os.path.basename(m)
    content = f.readlines()
    i = 0
    while i < len(content):
        if content[i] == "#originalArticleHeadline\n":
            s = ''
            i+=1
            while content[i].strip() != "#originalArticleBody":
                if '{"_id"' not in content[i].strip(): #there was some noise in the data like this 1558601614502.txt
                    s = s.strip() + ' ' + content[i].strip()
                i+=1
                if s.strip():
                    break
            df.at[j,'Sent'] = s
        elif content[i] == "#summaryHeadline\n":
            i+=1
            s = ''
            while content[i].strip() != "#summaryBody":
                s = s.strip() + ' ' + content[i].strip()
                i+=1
                if s.strip():
                    break
            df.at[j,'Positive'] = s
        i+=1
    j+=1
    # if j > 100:
    #     break

df = df[df['Sent'].apply(lambda d: len(d) > 1)]
df = df[df['Sent'] != '#originalArticleBody']
df = df.reset_index(drop=True)
# display(df)

ml = list(df['Sent'])
# for i in range(len(ml)):
for i in tqdm(range(len(ml)), desc='Lemmatizing and stopword removal',position=0, leave=True):
    ml[i] = lemmatize(ml[i],nlp).translate(str.maketrans('','',string.punctuation+"‟“’❝❞‚‘‛❛❜❟")).lower()
    ml[i] = get_nouns(ml[i],nlp)
    ml[i] = ' '.join([x for x in ml[i].split() if x not in hi_stopwords])
df['Sent_stripped'] = ml

for i in tqdm(range(len(df)), desc='Creating negative samples',position=0, leave=True):
# for i in range(len(df)):
    # print('Candidate sentence',df.loc[i,'Sent'])
    a, b, c = find_negative(df.loc[i,'Sent_stripped'],list(df['Sent_stripped']))
    # print('Negative one',df.loc[a,'Sent'],'\nNegative two', df.loc[b,'Sent'])
    # print('Negative:',df.loc[a,'Sent'])
    df.at[i,'Negative_recall'] = df.loc[a,'Sent']
    df.at[i,'Negative_precision'] = df.loc[b,'Sent']
    df.at[i,'Negative_fscore'] = df.loc[c,'Sent']

df.to_hdf('gdrive/MyDrive/SentSimHi/data/df3.h5',key='ids',mode='w') # this is a big dataset
# display(HTML(df[['Sent','Positive','Negative_recall','Negative_precision','Negative_fscore']].iloc[:3].to_html()))

df2 = pd.DataFrame([],columns=['fID','s1','s2','score'])
df['score'] = 1.0
df2 = pd.concat([df2,df[['fID','Sent','Positive','score']].rename(columns={'Sent':'s1','Positive':'s2'})])
df['score'] = 0.0
df2 = pd.concat([df2,df[['fID','Sent','Negative_fscore','score']].rename(columns={'Sent':'s1','Negative_fscore':'s2'})], ignore_index=True)
df2 = df2.sample(frac=1, random_state=123).reset_index(drop=True)
df2.to_hdf('gdrive/MyDrive/SentSimHi/data/sentence_pairs_38K.h5',key='ids',mode='w')
# display(HTML(df.iloc[:5].to_html()))
print(df2)