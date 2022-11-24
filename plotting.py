from matplotlib import pyplot as plt
# params = {'legend.fontsize': 22,
#           'legend.handlelength': 6}
# plt.rcParams.update(params)
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

font_size_num = 38
line_width_val = 5
font_name_val = 'Latin Modern Roman'

# SPC approach SR vs Test Acc vs Val Acc plots

fig1, ax = plt.subplots(1,1, figsize=(17,8))

# ax.boxplot([[3.3, 55.6, 21.9, 55.6, 21.9, 45.6, 33.0, 49.6, 17.8, 21.9] , [1.1, 42.6, 27.4, 42.6, 27.4, 43.7, 23.0, 26.7, 49.3, 32.2] , [46.7, 45.2, 5.6, 45.2, 5.6, 56.7, 45.2, 42.6, 53.3, 35.2]  , [65.2, 60.0, 58.1, 60.0, 58.1, 65.2, 62.6, 60.0, 61.9, 61.9], [64.4, 60.7, 58.1, 60.7, 58.1, 62.6, 53.3, 56.7, 65.6, 61.5]])
# ax.violinplot([[50.7, 23.3, 59.3, 43.7, 43.7, 22.2, 37.0, 21.9, 26.7, 2.2] , [51.1, 22.2, 52.2, 51.9, 38.9, 40.0, 39.6, 28.9, 55.2, 52.2], [47.8, 50.0, 11.9, 48.1, 42.6, 21.5, 50.7, 48.9, 51.9, 43.3], [49.3, 50.4, 10.7, 27.8, 41.5, 33.7, 49.6, 37.8, 44.1, 51.5], [40.0, 48.5, 13.7, 31.9, 40.4, 41.1, 35.9, 34.1, 45.6, 54.4]],showmeans=False, showmedians=False,showextrema=False)

# ax.boxplot([[31.9, 17.0, 23.0, 21.1, 5.6, 16.7, 25.6, 12.6, 18.5, 21.9] , [10.7, 16.3, 15.2, 7.0, 6.3, 10.4, 11.9, 7.4, 3.7, 6.3]     , [7.4, 11.9, 5.2, 8.9, 4.8, 0.7, 4.1, 5.2, 6.3, 4.4]         , [41.5, 55.6, 38.1, 37.0, 35.2, 46.7, 35.2, 5.2, 49.3, 43.7] , [47.4, 54.8, 48.1, 51.9, 46.7, 41.9, 39.6, 14.4, 59.6, 40.4]])
# ax.violinplot([[98.0, 98.2, 98.2, 98.5, 97.4, 97.6, 96.8, 98.3, 98.5, 98.1], [98.6, 96.1, 98.8, 99.0, 97.2, 98.1, 98.3, 98.8, 98.0, 98.8], [98.8, 99.0, 97.4, 98.9, 98.3, 98.8, 98.2, 98.9, 98.5, 99.0], [99.1, 99.2, 99.0, 98.7, 98.9, 98.9, 98.9, 98.8, 98.6, 98.8], [97.9, 99.2, 98.9, 99.0, 98.9, 98.6, 98.7, 99.0, 99.1, 99.1]])

# ax.boxplot([[55.6, 55.9, 57.8, 54.1, 60.0, 57.0, 54.1, 55.9, 46.7, 57.4], [56.3, 52.2, 55.9, 53.0, 58.9, 54.8, 54.8, 58.9, 52.2, 58.1], [53.3, 54.1, 45.9, 47.4, 51.5, 38.9, 43.3, 53.3, 49.3, 56.3], [62.6, 61.9, 62.2, 62.2, 63.7, 66.3, 62.2, 60.0, 60.0, 64.4], [63.7, 56.7, 58.1, 59.6, 60.4, 64.4, 61.1, 60.4, 57.8, 60.4], [60.7, 54.1, 50.7, 53.3, 55.9, 58.1, 60.0, 58.9, 57.0, 51.1], [57.4, 55.2, 37.0, 49.3, 52.2, 55.9, 52.6, 58.9, 53.7, 53.7], [52.2, 53.0, 45.6, 42.6, 45.9, 54.8, 51.9, 57.4, 51.5, 49.6], [48.9, 48.9, 34.8, 50.4, 43.7, 53.3, 51.1, 58.1, 47.8, 47.4]])
# ax.violinplot([[55.6, 55.9, 57.8, 54.1, 60.0, 57.0, 54.1, 55.9, 46.7, 57.4], [56.3, 52.2, 55.9, 53.0, 58.9, 54.8, 54.8, 58.9, 52.2, 58.1], [53.3, 54.1, 45.9, 47.4, 51.5, 38.9, 43.3, 53.3, 49.3, 56.3], [62.6, 61.9, 62.2, 62.2, 63.7, 66.3, 62.2, 60.0, 60.0, 64.4], [63.7, 56.7, 58.1, 59.6, 60.4, 64.4, 61.1, 60.4, 57.8, 60.4], [60.7, 54.1, 50.7, 53.3, 55.9, 58.1, 60.0, 58.9, 57.0, 51.1], [57.4, 55.2, 37.0, 49.3, 52.2, 55.9, 52.6, 58.9, 53.7, 53.7], [52.2, 53.0, 45.6, 42.6, 45.9, 54.8, 51.9, 57.4, 51.5, 49.6], [48.9, 48.9, 34.8, 50.4, 43.7, 53.3, 51.1, 58.1, 47.8, 47.4]],showmeans=True, showmedians=True,showextrema=False)



# ax.set_ylim(bottom = 0, top=110)
# # plt.title('FC3')

# plt.show()
# exit()

# ax.fill_between([1,2,3,4,5],[44.4 + 18.8, 41.8 + 17.0, 36.6 + 21.4, 36.8 + 22.6, 34.1 + 18.1], [44.4 - 18.8, 41.8 - 17.0, 36.6 - 21.4, 36.8 - 22.6, 34.1 - 18.1],alpha=0.2,color=(0.0, 0.0, 1.0))
# ax.plot([1,2,3,4,5],[44.4, 41.8, 36.6, 36.8, 34.1],color=(0.0,0.0,1.0),marker='s', markersize=10,label=r'$SR_{+rn}$')

# ax.fill_between([1,2,3,4,5],[13.7 + 8.6, 10.3 + 1.4, 7.4  + 1.6 , 5.9  + 2.8 , 8.2  + 1.3 ], [13.7 - 8.6, 10.3 - 1.4, 7.4  - 1.6 , 5.9  - 2.8 , 8.2  - 1.3 ],alpha=0.2,color=(1.0, 0.0, 0.0))
# ax.plot([1,2,3,4,5],[13.7, 10.3, 7.4 , 5.9 , 8.2 ],color=(1.0,0.0,0.0),marker='s', markersize=10,label=r'$SR_{-rn}$')

# ax.fill_between([1,2,3,4,5],[98.3 + 0.3, 98.8 + 0.2, 98.8 + 0.2, 99.0 + 0.2, 98.9 + 0.4], [98.3 - 0.3, 98.8 - 0.2, 98.8 - 0.2, 99.0 - 0.2, 98.9 - 0.4],alpha=0.2,color=(0.0, 0.0, 1.0))
# ax.plot([1,2,3,4,5],[98.3, 98.8, 98.8, 99.0, 98.9],color=(0.0,0.0,1.0),marker='^', markersize=10,label=r'$Test\ Accuracy_{+rn}$')

# ax.fill_between([1,2,3,4,5],[89.1 + 2.2, 94.0 + 0.6, 95.1 + 0.6, 95.1 + 0.5, 95.0 + 0.7],[89.1 - 2.2, 94.0 - 0.6, 95.1 - 0.6, 95.1 - 0.5, 95.0 - 0.7],alpha=0.2,color=(1.0, 0.0, 0.0))
# ax.plot([1,2,3,4,5],[89.1, 94.0, 95.1, 95.1, 95.0],color=(1.0,0.0,0.0),marker='^', markersize=10,label=r'$Test\ Accuracy_{-rn}$')








# ax.fill_between([1,2,3],[110, 110, 110],[0,0,0], alpha=0.2, color=(0.59, 0.60, 0.60))
# ax.fill_between([1,2,3,4,5,6,7,8,9],[55.5 + 3.5, 55.5 + 2.6, 49.3 + 5.5, 62.6 + 1.9, 60.3 + 2.4, 56.0 + 3.6, 52.6 + 6.1, 50.5 + 4.6, 48.4 + 6.1], [55.5 - 3.5, 55.5 - 2.6, 49.3 - 5.5, 62.6 - 1.9, 60.3 - 2.4, 56.0 - 3.6, 52.6 - 6.1, 50.5 - 4.6, 48.4 - 6.1],alpha=0.2,color=(0.0, 0.0, 1.0))
# ax.plot([1,2,3,4,5,6,7,8,9],[55.5, 55.5, 49.3, 62.6, 60.3, 56.0, 52.6, 50.5, 48.4],color=(0.0,0.0,1.0),marker='s', markersize=10,label='First two layers frozen in SPC encoder')

# ax.text(1.1, 100, 'Fine-tuning on\nout-domain (Inshorts) dataset',fontsize=17)

# ax.fill_between([3,4,5,6,7,8,9],[110,110, 110,110,110, 110,110],[0,0,0,0,0,0,0], alpha=0.2, color=(0.15, 0.70, 0.38))
# ax.fill_between([1,2,3,4,5,6,7,8,9],[56.6 + 3.4, 55.3 + 4.0, 48.7 + 7.1, 61.5 + 2.1, 60.0 + 2.4, 56.2 + 3.7, 51.3 + 4.9, 47.5 + 9.3, 43.4 + 12.6], [56.6 - 3.4, 55.3 - 4.0, 48.7 - 7.1, 61.5 - 2.1, 60.0 - 2.4, 56.2 - 3.7, 51.3 - 4.9, 47.5 - 9.3, 43.4 - 12.6],alpha=0.2,color=(1.0, 0.0, 0.0))
# ax.plot([1,2,3,4,5,6,7,8,9],[56.6, 55.3, 48.7, 61.5, 60.0, 56.2, 51.3, 47.5, 43.4],color=(1.0,0.0,0.0),marker='s', markersize=10,label='No layer frozen in SPC encoder')
# ax.text(3.05, 100, 'Fine-tuning on\nin-domain (Asha_Qs) dataset',fontsize=17)

clist = [( 176/255, 58/255, 46/255 ), ( 136/255, 78/255, 160/255 ), ( 46/255, 134/255, 193/255 ), ( 212/255, 172/255, 13/255 ), ( 211/255, 84/255, 0 )]

ax.fill_between([1,2,3],[110, 110, 110],[0,0,0], alpha=0.2, color=(0.59, 0.60, 0.60))
ax.fill_between([3,4,5,6,7,8,9],[110,110, 110,110,110, 110,110],[0,0,0,0,0,0,0], alpha=0.2, color=(0.15, 0.70, 0.38))
ax.text(1.2, 70, '        Fine-tuning on\nout-domain (Inshorts) dataset',fontsize=16,bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 10})
ax.text(4.05, 70, '        Fine-tuning on\nin-domain (AshaQs) dataset',fontsize=16,bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 10})


ax.fill_between([1,2,3,4,5,6,7,8,9],[32.0 + 2.3, 31.8 + 1.7, 25.5 + 4.5, 37.3 + 1.2, 35.7 + 2.3, 31.7 + 3.2, 28.5 + 4.8, 27.3 + 3.6, 25.5 + 4.4], [32.0 - 2.3, 31.8 - 1.7, 25.5 - 4.5, 37.3 - 1.2, 35.7 - 2.3, 31.7 - 3.2, 28.5 - 4.8, 27.3 - 3.6, 25.5 - 4.4],alpha=0.2,color=clist[0])
ax.plot([1,2,3,4,5,6,7,8,9],[32.0, 31.8, 25.5, 37.3, 35.7, 31.7, 28.5, 27.3, 25.5],color=clist[0],marker='s', markersize=10,label='mAP')

ax.fill_between([1,2,3,4,5,6,7,8,9],[45.0 + 3.1, 44.9 + 2.1, 37.9 + 6.0, 51.6 + 1.6, 49.5 + 2.6, 45.0 + 3.5, 41.2 + 5.9, 39.8 + 4.2, 37.6 + 5.6], [45.0 - 3.1, 44.9 - 2.1, 37.9 - 6.0, 51.6 - 1.6, 49.5 - 2.6, 45.0 - 3.5, 41.2 - 5.9, 39.8 - 4.2, 37.6 - 5.6],alpha=0.2,color=clist[1])
ax.plot([1,2,3,4,5,6,7,8,9],[45.0, 44.9, 37.9, 51.6, 49.5, 45.0, 41.2, 39.8, 37.6],color=clist[1],marker='o', markersize=10,label='MRR')

ax.fill_between([1,2,3,4,5,6,7,8,9],[55.5 + 3.5, 55.5 + 2.6, 49.3 + 5.5, 62.6 + 1.9, 60.3 + 2.4, 56.0 + 3.6, 52.6 + 6.1, 50.5 + 4.6, 48.4 + 6.1], [55.5 - 3.5, 55.5 - 2.6, 49.3 - 5.5, 62.6 - 1.9, 60.3 - 2.4, 56.0 - 3.6, 52.6 - 6.1, 50.5 - 4.6, 48.4 - 6.1],alpha=0.2,color=clist[2])
ax.plot([1,2,3,4,5,6,7,8,9],[55.5, 55.5, 49.3, 62.6, 60.3, 56.0, 52.6, 50.5, 48.4],color=clist[2],marker='^', markersize=10,label='SR')

ax.fill_between([1,2,3,4,5,6,7,8,9],[47.5 + 3.2, 47.5 + 2.1, 40.6 + 5.7, 54.2 + 1.5, 52.0 + 2.4, 47.6 + 3.4, 43.9 + 5.9, 42.4 + 4.2, 40.1 + 5.6], [47.5 - 3.2, 47.5 - 2.1, 40.6 - 5.7, 54.2 - 1.5, 52.0 - 2.4, 47.6 - 3.4, 43.9 - 5.9, 42.4 - 4.2, 40.1 - 5.6],alpha=0.2,color=clist[3])
ax.plot([1,2,3,4,5,6,7,8,9],[47.5, 47.5, 40.6, 54.2, 52.0, 47.6, 43.9, 42.4, 40.1],color=clist[3],marker='D', markersize=10,label='nDCG')

ax.fill_between([1,2,3,4,5,6,7,8,9],[27.8 + 1.9, 27.5 + 1.6, 22.5 + 3.0, 32.6 + 0.9, 31.1 + 1.8, 27.6 + 3.0, 25.3 + 4.1, 23.8 + 3.4, 22.6 + 3.9], [27.8 - 1.9, 27.5 - 1.6, 22.5 - 3.0, 32.6 - 0.9, 31.1 - 1.8, 27.6 - 3.0, 25.3 - 4.1, 23.8 - 3.4, 22.6 - 3.9],alpha=0.2,color=clist[4])
ax.plot([1,2,3,4,5,6,7,8,9],[27.8, 27.5, 22.5, 32.6, 31.1, 27.6, 25.3, 23.8, 22.6],color=clist[4],marker='X', markersize=10,label='P@k')


# ax.fill_between([1,2,3,4,5],[98.0 + 0.5, 98.2 + 0.9, 98.6 + 0.5, 98.9 + 0.2, 98.8 + 0.4], [98.0 - 0.5, 98.2 - 0.9, 98.6 - 0.5, 98.9 - 0.2, 98.8 - 0.4],alpha=0.2,color=(0.0, 0.0, 1.0))
# ax.plot([1,2,3,4,5],[98.0, 98.2, 98.6, 98.9, 98.8],color=(0.0,0.0,1.0),marker='^', markersize=10,label=r'$Test\ Accuracy_{+rn}$')

# ax.fill_between([1,2,3,4,5],[92.2 + 0.5, 94.0 + 0.9, 95.7 + 0.2, 93.6 + 2.1, 96.0 + 0.2],[92.2 - 0.5, 94.0 - 0.9, 95.7 - 0.2, 93.6 - 2.1, 96.0 - 0.2],alpha=0.2,color=(1.0, 0.0, 0.0))
# ax.plot([1,2,3,4,5],[92.2, 94.0, 95.7, 93.6, 96.0],color=(1.0,0.0,0.0),marker='^', markersize=10,label=r'$Test\ Accuracy_{-rn}$')



ax.set_xlabel("Epochs",fontsize=font_size_num,fontname=font_name_val)
ax.set_ylim(bottom = 15, top=80)
ax.grid(axis = 'y')
# ax.legend()
plt.xticks(fontsize=font_size_num,fontname=font_name_val)
plt.yticks(fontsize=font_size_num,fontname=font_name_val)
L = plt.legend(fontsize=22, handlelength=3, loc='upper center', bbox_to_anchor=(0.5, 1.25),ncol=5, fancybox=True, shadow=True)
# plt.setp(L.texts, family='DejaVu Sans')
plt.setp(L.texts, family=font_name_val)

plt.show()

exit()

df = pd.read_csv('/home/ritwik/git/AshaQnA/asha_data/GradData_run_ID_180.csv')

# plt.rc('font',family=font_name_val) # this changes all the fonts

type_l = df['type'].unique()

layer_nums = [1,6,12]
layers = ['model.embeddings.word_embeddings.weight']+['model.encoder.layer.'+str(x-1)+'.attention.self.value.weight' for x in layer_nums]+['fc3.weight']
layers = ['model.encoder.layer.'+str(x-1)+'.attention.self.value.weight' for x in layer_nums]+['fc3.weight']
name_layers = ['embedding layer']+['layer '+str(x) for x in layer_nums]+['fc3']
name_layers = ['layer '+str(x) for x in layer_nums]+['fc3']
line_styles = [(0,(3,0.5)),(0,(6,0.5)),(0,(5,0.5,2,0.5)),(0,(2,0.5)),(0,(5,0.5,1,0.5)),(0,(3,0.25,1,0.25,1,0.25)),'solid',(0,(5,0.25,1,0.25,1,0.25,1,0.25))]
vals = []

# model.bert.encoder.layer.0.attention.self.value.weight

fig1, ax = plt.subplots(1,1, figsize=(17,8))
ax.set_yscale('log')

for l in layers:
	df = df[df['type'] == 'l2_norm'].reset_index(drop=True)
	vals.append(list(df[l]))

for x in range(552,len(vals[0]),552):
	ax.plot([x,x], [0,5], color='black',linewidth=2,alpha=0.1)

for v,n,ls in zip(vals, name_layers, line_styles):
	v = gaussian_filter1d(v, sigma=4)
	ax.plot(np.arange(len(v)),v,label=n,alpha=0.5,linewidth=line_width_val,linestyle=ls)

# ax.tick_params(axis='both', which='major', labelsize=10)

plt.xticks(fontsize=font_size_num,fontname=font_name_val)
plt.yticks(fontsize=font_size_num,fontname=font_name_val)

plt.xlabel("Iterations",fontsize=font_size_num,fontname=font_name_val)
plt.ylabel("(smoothed) L2 norm of gradient",fontsize=font_size_num-4,fontname=font_name_val)
plt.ylim(bottom = 0.0001, top=10)
# L = plt.legend(fontsize=30, handlelength=6)
L = plt.legend(fontsize=25, handlelength=4, loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=4, fancybox=True, shadow=True)
plt.setp(L.texts, family='DejaVu Sans')
plt.show()
exit()

fig1, ax = plt.subplots(1,1, figsize=(17,8))

file = open('/home/ritwik/train_hold_out_metrics.txt','r')
content = file.readlines()
file.close()
vals = [100*float(line.strip().split()[4][7:]) for line in content]
for x in range(24,len(vals),24):
	ax.plot([x,x], [0,90], color='black',linewidth=2,alpha=0.1)
ax.plot(np.arange(len(vals)), vals, label='Hold_out_test SR',alpha=0.5,linewidth=line_width_val,linestyle='solid')

file = open('/home/ritwik/train_test_acc.txt','r')
content = file.readlines()
file.close()
vals = [float(line.strip().split()[2]) for line in content]
ax.plot(np.arange(len(vals)), vals, label='Test acc',alpha=0.5,linewidth=line_width_val,linestyle=(0,(3,0.5)))

file = open('/home/ritwik/train_val_loss_acc.txt','r')
content = file.readlines()
file.close()
vals = [float(line.strip().split()[3]) for line in content]
ax.plot(np.arange(len(vals)), vals, label='Val acc',alpha=0.5,linewidth=line_width_val,linestyle=(0,(3,0.25,1,0.25,1,0.25)))

# fig1, ax = plt.subplots(1,1, figsize=(17,8))

# x = [10,20,30,40,50]
# y = [30,30,30,30,30]
  
# # plot lines
# plt.plot(x, y, label = "line 1", linestyle=(2, (6,5,5,6,4,7,5,6,6,5)))
# plt.plot(y, x, label = "line 2")
plt.xticks(fontsize=font_size_num,fontname=font_name_val)
plt.yticks(fontsize=font_size_num,fontname=font_name_val)

plt.xlabel("Iterations",fontsize=font_size_num,fontname=font_name_val)
plt.legend(fontsize=26, handlelength=6)
plt.ylim(bottom = 10, top=110)
plt.show()
