import pandas as pd
import editdistance
import string
# pd.set_option('display.max_rows', 50000)
# pd.set_option('display.max_columns', 30)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 5)

import pickle
from indictrans import Transliterator
import stanza
import re
from tqdm import tqdm
trn = Transliterator(source='eng', target='hin')

# 7 साल का बच्चा बेड पे  बाथरूम कर देता हैं. इसके लिए क्या करें?
# 7 साल का बच्चा बेड पर बाथरूम कर देता है तो इसके लिए हम क्या करें

# print(editdistance.eval("हैं क्या करें","है तो हम करें"))
# df = pd.read_csv('/home/ritwik/git/AshaQnA/Outside_questions_ground_truth.csv')

# print(len(set(df[df['Final score']!='0']['User_query'])))
# print(sum(df['Final score']=='0'))
# exit()

def clean(s):
	return re.sub(r'\s+',' ',s).strip()

nlp = stanza.Pipeline('hi', use_gpu=False)

def lemmatize(s, nlp):
    s = s.replace(',','').replace('!','').replace('  ',' ')
    doc = nlp(s)
    ml = []
    for sentence in doc.sentences:
        for word in sentence.words:
            ml.append(word.lemma)
    s = ' '.join(ml)
    return s

f = open('/home/ritwik/git/AshaQnA/hi_stopwords.pkl','rb')
stp = pickle.load(f)
stp = set(stp)
f.close()

def fetch_most_similar(allQnA,q,n):
	q = lemmatize(q,nlp)
	q = (set(q.split(' ')) - stp)
	allQnA['common_words'] = allQnA['lemmaQuestion'].apply(lambda d: len((set(d.split()) - stp).intersection(q)))
	allQnA = allQnA.sort_values('common_words').reset_index(drop=True)
	return list(allQnA.iloc[n:,2])

df = pd.read_csv('/home/ritwik/git/AshaQnA/other_codes/Metrics/Data/Relevance.csv')
# c = []
# d = []
# for _, row in df.iterrows():
# 	c.append(len(set(list(row))))
# 	d.append(editdistance.eval(row['Questions'].translate(str.maketrans('','',string.punctuation+"‟“’❝❞‚‘‛❛❜❟")).lower().replace('|','').replace('।',''), row['RQ1'].translate(str.maketrans('','',string.punctuation+"‟“’❝❞‚‘‛❛❜❟")).lower().replace('|','').replace('।','')))
# df.insert(0,'counts',c)
# df.insert(1,'leven_dist',d)
# df = df[df['leven_dist']>10]
# df = df.sort_values(['RQ1','counts'])
# df = df.drop_duplicates(subset=['RQ1'], keep='last').reset_index(drop=True)
# df.to_csv('/home/ritwik/git/AshaQnA/other_codes/Metrics/Data/Relevance_temp.csv',index=None)
# exit()

df.fillna('',inplace=True)
cols = df.columns

allQnA = pd.read_csv('/home/ritwik/git/AshaQnA/allQnA.csv')

data = []

for _, row in tqdm(df.iterrows(),total=len(df)):
	q = clean(row['Questions'])
	if row['RQ1'] == 'NF':
		continue
		for l in fetch_most_similar(allQnA,q,n=-3):
			data.append([q,l,0,''])
	elif False: # row['RQ3'] == '':
		count = 0
		for l in fetch_most_similar(allQnA,q,n=-5):
			if l!=row['RQ1'] and l!=row['RQ2']:
				data.append([q,l,0,''])
				count+=1
				if count==3:
					break
		if row['RQ2']!='':
			data.append([q,row['RQ2'].strip('X').strip(), 1 if row['RQ2'][-1]=='X' else 2,''])
		data.append([q,row['RQ1'].strip('X').strip(), 1 if row['RQ1'][-1]=='X' else 2,''])
	else:
		for c in cols[2:]:
			if row[c] == '':
				break
			data.append([q,clean(row[c].strip('X')),1 if row[c][-1]=='X' else 2,''])


df2 = pd.DataFrame(data,columns=['User_query','Database_question','Human_score','Expert_score'])

# df2 = pd.read_csv('/home/ritwik/git/Metrics/Data/Relevance_for_kirti.csv')

df3 = pd.read_excel('/home/ritwik/git/AshaQnA/other_codes/Metrics/Data/Annotations_outer.xlsx')
df3['User_Question'] = df3['User_Question'].apply(lambda d: clean(d))

# print(df3)
data = []
ql = list(df2['User_query'].unique())
# ql = list(df3['User_Question'].unique())

for q in tqdm(ql):
	q = clean(q)
	tdf = df3[df3['User_Question']==q]
	if len(tdf) > 0:
		for _, row in tdf.iterrows():
			for i in range(3):
				m = min(list(row[['prerna_'+str(i+1),	'nitika_'+str(i+1),	'mrinalini_'+str(i+1)]]))
				if m in [1,2]:
					data.append([q,clean(row['OPTION_'+str(i+1)]), 2 if m==1 else 1, ''])

df3 = pd.DataFrame(data,columns=['User_query','Database_question','Human_score','Expert_score'])
df2 = pd.concat([df2,df3])
df2 = df2.drop_duplicates(subset=['User_query','Database_question'], keep='first').sort_values('User_query')



df2.to_csv('/home/ritwik/git/AshaQnA/other_codes/Metrics/Data/Relevance_for_kirti.csv',index=None)
# print(df2)
