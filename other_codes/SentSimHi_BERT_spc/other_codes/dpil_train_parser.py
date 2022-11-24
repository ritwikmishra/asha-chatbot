import csv
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

tree = ET.parse('HindiTask1.xml')
root = tree.getroot()

data = []
ml = []

for child in tqdm(root):
	# print(child.tag, child.attrib)
	# input('1')
	for child2 in child:
		# print(child2.tag, child2.attrib)
		# input('2')
		for child3 in child2:
			ml.append(child3.text)
		if ml:
			data.append(ml+[child2.attrib['pID']])
			ml = []

df = pd.DataFrame(data, columns=['Sent1','Sent2','y','pID'])
df.to_csv('dpil_train_hi.csv',index=None)

