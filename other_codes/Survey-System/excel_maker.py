import pandas as pd
from tqdm import tqdm

df = pd.read_csv('/home/ritwik/git/Survey-System/final-Ques_rand.csv')

dfp = pd.read_csv('/home/ritwik/git/Survey-System/response/prerna.csv')
dfn = pd.read_csv('/home/ritwik/git/Survey-System/response/nitika.csv')
dfm = pd.read_csv('/home/ritwik/git/Survey-System/response/mrinalini.csv')
dfp = dfp[dfp['Ans']!=0]
dfn = dfn[dfn['Ans']!=0]
dfm = dfm[dfm['Ans']!=0]

dfa = [dfp, dfn, dfm]

df = df.drop(columns=['SCORE_1','SCORE_2','SCORE_3'])

df.insert(6,'prerna_1','')
df.insert(7,'nitika_1','')
df.insert(8,'mrinalini_1','')

df.insert(11,'prerna_2','')
df.insert(12,'nitika_2','')
df.insert(13,'mrinalini_2','')

df.insert(16,'prerna_3','')
df.insert(17,'nitika_3','')
df.insert(18,'mrinalini_3','')

print(df.columns)
# dfa = [pd.read_csv('/home/ritwik/git/Survey-System/response/prerna.csv'), pd.read_csv('/home/ritwik/git/Survey-System/response/nitika.csv'), pd.read_csv('/home/ritwik/git/Survey-System/response/mrinalini.csv')]

# for d in dfa:
# 	for _,x in d.iterrows():

i = 1
for ((_,p),(_,n),(_,m)) in tqdm(zip(dfp.iterrows(), dfn.iterrows(), dfm.iterrows())):
	pg = int((int(p['Ques']) - 1 )/3 + 1)
	df.at[pg-1,'prerna_'+str(i)] = p['Ans']
	pg = int((int(n['Ques']) - 1 )/3 + 1)
	df.at[pg-1,'nitika_'+str(i)] = n['Ans']
	pg = int((int(m['Ques']) - 1 )/3 + 1)
	df.at[pg-1,'mrinalini_'+str(i)] = m['Ans']
	i+=1
	if (i==4):
		i = 1

print(df)

df.to_csv('temp.csv',index=None)

