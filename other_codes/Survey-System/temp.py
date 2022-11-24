import pandas as pd 
import random
import string

def r():
	return ''.join(random.choices('ऄअ आइईउ ऊऋऌऍऎ एऐऑऒ ओऔक खगघङ चछजझ ञटठडढ णतथदध नऩपफब भमयरऱल ळऴवश षसहऺऻ़ क़ख़ग़ज़ ड़ढ़फ़य़', k = 100)) 

data = []
resp = ['Totally relevant', 'Somewhat relevant','Irrelevant']

number_of_user_questions = nuq = 500
models_to_compare = mtc = 4
top_k = 3
done = 10

cols = []
for i in range(top_k):
	cols.append('OPTION_'+str(i+1))
	cols.append('EXP_pred_'+str(i+1))

k = 1
for i in range(nuq):
	for j in range(mtc):
		row = [k]
		random.seed(i+1) 
		row.append('User_Question '+str(i+1)+': '+r())
		row.append('Model name '+str(j+1))
		for n in range(top_k):
			random.seed(k)
			row.append('option '+str(n+1)+': '+r())
			if k < done:
				row.append(random.choices(resp)[0])
			else:
				row.append('')
		data.append(row)
		k+=1

df = pd.DataFrame(data, columns=['Page','User Question','Model name']+cols)
df.to_csv('temp.csv',index=None)
print(df)
