# this code takes input as different hold_out_metrices
# output will be std and mean
import statistics as st

mylist = [(112,114),(195,201),(270,279)] # for fc1 random sents on
# mylist = [(121,123),(188,194),(260,269)] # for fc2 random sents on
mylist = [(124,126),(181,187),(280,289)] # for fc3 random sents on

mylist = [(115,117),(304,310)] # for fc1 random sents off
# mylist = [(136,138),(202,208)] # for fc2 random sents off
mylist = [(166,168),(297,303)] # for fc3 random sents off

# mylist = [(356,360)] # for fc1 with E frozen
# mylist = [(361,365)] # for fc2 with E frozen
# mylist = [(366,370)] # for fc3 with E frozen

# mylist = [(311,315)] # for fc1 with EL0 frozen
# mylist = [(316,320),(371,375)] # for fc2 with EL0 frozen
# mylist = [(321,325),(376,379)] # for fc3 with EL0 frozen

# mylist = [(326,330)] # for fc1 with EL0L1 frozen
# mylist = [(331,335)] # for fc2 with EL0L1 frozen
# mylist = [(336,340)] # for fc3 with EL0L1 frozen

# mylist = [(341,345)] # for fc1 with EL0L1L2 frozen
# mylist = [(346,350)] # for fc2 with EL0L1L2 frozen
# mylist = [(351,355)] # for fc3 with EL0L1L2 frozen

# mylist = [(381,385), (427,430)] # fc2 with inshorts data
# mylist = [(401,403)] # fc3 after inshorts data with EL0 frozen <-----------------


# mylist = [(386,390)] # fc2 after inshorts data <-----------------
# mylist = [(407,411)] # fc2 after inshorts data with E frozen
# mylist = [(396,400),(422,426)] # fc2 after inshorts data with EL0 frozen <-----------------
# mylist = [(412,416)] # fc2 after inshorts data with EL0L1 frozen
# mylist = [(417,421)] # fc2 after inshorts data with EL0L1L2 frozen

# mylist = [(431,435)] # fc3 after inshorts data with E frozen
# mylist = [(401,403),(446,450)] # fc3 after inshorts data with EL0 frozen
# mylist = [(436,440)] # fc3 after inshorts data with EL0L1 frozen
# mylist = [(441,445)] # fc3 after inshorts data with EL0L1L2 frozen
# mylist = [(451,454)] # fc3 with inshorts data

# mylist = [(455,457)] # fc2 with only asha data
# mylist = [(458,460)] # fc2 with only asha data and negs repeated twice (matlab posx1 negsx2)
# mylist = [(461,463)] # fc2 with only asha data and EL0 frozen

# mylist = [(391,395),(464,468)] # fc2 after inshorts data with best model
# mylist = [(436,440),(469,473)] # fc3 after inshorts with EL0L1 frozen
# mylist = [(401,403),(446,450),(474,475)] # fc3 after inshorts with EL0 frozen

# mylist = [(476,485)] # fc2 after inshorts data with best model with EL0L1 frozen
# mylist = [(486,495)] # fc2 after inshorts data with best model with EL0 frozen

# mylist = [(496,505)] # fc1 after inshorts data with best model
# mylist = [(506,515)] # fc3 after inshorts data with best model

# mylist = [(516,525)] # fc1 after inshorts with EL0 frozen

mylist = [(401,403),(446,450),(474,475)] # fc3 after inshorts with EL0 frozen
# mylist = [(526,535)] # fc3 after inshorts with EL0 frozen with inshorts random negs off
# mylist = [(537,546)] # fc3 after inshorts data with nothing frozen

# mylist = [(556,558)] # best model (fc3AfterInshortsEL0frozen: faief) with hingeloss: kharab

# mylist = [(547,549),(559,565)] # best model (fc3AfterInshortsEL0frozen: faief) with cls
# mylist = [(566,575)] # best model (fc3AfterInshortsEL0frozen: faief) with cls with nothing frozen

# mylist = [(550,552),(576,577)] # best model (fc3AfterInshortsEL0frozen: faief) with mbert : better
# mylist = [(553,555), (578,579)] # best model (fc3AfterInshortsEL0frozen: faief) with bcewithlogits : better


mylist = [(580,589)] # pooler fc3 after inshorts EL0 frozen but with mbert
# mylist = [(590,599)] # pooler fc3 after inshorts nothing frozen but with mbert
# mylist = [(600,609)] # cls fc3 after inshorts EL0 frozen but with mbert
# mylist = [(610,619)] # cls fc3 after inshorts nothing frozen but with mbert

# all mbert (cased) below
# mylist = [(620,624),(696,700)] # pooler fc3 only_asha EL0 frozen
# mylist = [(625,629)] # pooler fc3 only_asha nothing frozen
# mylist = [(630,634),(701,705)] # pooler fc3 only inshorts EL0 frozen

# mylist = [(635,639)] # pooler fc3 only inshorts nothing frozen
# mylist = [(640,644),(706,710)] # pooler fc2 after inshorts EL0 frozen
# mylist = [(644,649),(711,715)] # pooler fc1 after inshorts EL0 frozen
mylist = [(771,780)] # pooler fc4 after inshorts EL0 frozen

# mylist = [(650,654)] # pooler fc3 after inshorts data EL0 frozen but with mbert uncased
# mylist = [(655,659),(716,720)] # pooler fc3 after inshorts data EL0 frozen but with indic bert
# mylist = [(660,664)] # pooler fc3 after inshorts data EL0 frozen but with xlm 1280
# mylist = [(665,669)] # pooler fc3 after inshorts data EL0 frozen but with xlm 1024
# mylist = [(670,674)] # pooler fc3 after inshorts data EL0 frozen but with xlm xnli 1024 

# mylist = [(675,679),(721,725)] # pooler fc3 after inshorts data only E frozen with mbert cased
# mylist = [(680,684),(726,730)] # pooler fc3 after inshorts data EL0L1 frozen with mbert cased
# mylist = [(685,689),(731,735)] # pooler fc3 after inshorts data EL0L1L2 frozen with mbert cased
# mylist = [(690,694),(736,740)] # pooler fc3 after inshorts data half bert frozen with mbert cased

# mylist = [(741,745),(756,760)] # pooler fc3 after 1 epoch inshorts EL0 frozen but with mbert
# mylist = [(746,750),(761,765)] # pooler fc3 after 2 epoch inshorts EL0 frozen but with mbert
# mylist = [(751,755),(766,770)] # pooler fc3 after 4 epoch inshorts EL0 frozen but with mbert

mylist = [(901,905)] # inshorts+dpil
mylist = [(851,855)] # only dpil
mylist = [(861,865)] # only dpil nothing frozen
mylist = [(871,875)] # only dpil with cls
mylist = [(886,890)] # only dpil with cls nothing frozen

mylist = [(876,880)] # inshorts then dpil
mylist = [(896,900)] # only inshorts


# # rand negs off
mylist = [(906,910)] # inshorts+dpil
mylist = [(856,860)] # only dpil
mylist = [(866,870)] # only dpil nothing frozen
# mylist = [(881,885)] # only dpil with cls
# mylist = [(891,895)] # only dpil with cls nothing frozen
# mylist = [(916,920)] # inshorts then dpil
# mylist = [(911,915)] # only inshorts

print(mylist)
a = input('1 Accuracy?\n2 SR?\n3 Val loss\n4 Train loss\t')
if a  == '2':
	ml = {}
	metrics = {}
	pth = '/media/data_dump/Ritwik/git/AshaQnA/data/train_hold_out_metrics_'
	linelist = []
	for (start, end) in mylist:
		for i in range(start, end+1):
			file = open(pth+str(i)+'.txt')
			content = file.readlines()
			file.close()
			linelist += content
	linelist += ['exit']
	for i, m in enumerate(['map:','mrr:','avg_sr:','avg_ndcg:','avg_p3:']):
		for line in linelist:
			if line == 'exit':
				break
			line = line.split()
			if line[0] not in ml.keys():
				tl = []
				ml[line[0]] = tl
			else:
				tl = ml[line[0]]
			tl.append(round(round(float(line[i+2].replace(m,'')),3)*100,3))
			ml[line[0]] = tl
		metrics[m] = ml
		ml = {}
	for m in ['map:','mrr:','avg_sr:','avg_ndcg:','avg_p3:']:
		print(m)
		ml = metrics[m]
		for k in ml.keys():
			print(k, round(st.mean(ml[k]),1), '±', round(st.stdev(ml[k]),1), ml[k])
elif a == '1':
	ml = {}
	pth = '/media/data_dump/Ritwik/git/AshaQnA/data/train_test_acc_'
	linelist = []
	for start, end in mylist:
		for i in range(start, end+1):
			file = open(pth+str(i)+'.txt')
			content = file.readlines()
			file.close()
			linelist += content
	linelist += ['exit']
	for line in linelist:
		if line == 'exit':
			break
		line = line.split()
		if line[0] not in ml.keys():
			tl = []
			ml[line[0]] = tl
		else:
			tl = ml[line[0]]
		tl.append(round(100*float(line[-1]),1) if round(100*float(line[-1]),1) <=100 else round(float(line[-1]),1))
		ml[line[0]] = tl

	for k in ml.keys():
		print(k, round(st.mean(ml[k]),1), '±', round(st.stdev(ml[k]),1), ml[k])
elif a =='3':
	ml = {}
	pth = '/media/data_dump/Ritwik/git/AshaQnA/data/train_val_loss_acc_'
	linelist = []
	for start, end in mylist:
		for i in range(start, end+1):
			file = open(pth+str(i)+'.txt')
			content = file.readlines()
			file.close()
			linelist += content
	linelist += ['exit']
	for line in linelist:
		if line == 'exit':
			break
		line = line.split()
		if line[0] not in ml.keys():
			tl = []
			ml[line[0]] = tl
		else:
			tl = ml[line[0]]
		tl.append(round(float(line[2]),5))
		ml[line[0]] = tl

	for k in ml.keys():
		print(k, round(st.mean(ml[k]),5), '±', round(st.stdev(ml[k]),5), ml[k])
	
	ml = {}
	b = input('Any specific experiment runID?')
	file = open(pth+str(b)+'.txt')
	content = file.readlines()
	file.close()
	linelist = content + ['exit']

	for line in linelist:
		if line == 'exit':
			break
		line = line.split()
		if line[0] not in ml.keys():
			tl = []
			ml[line[0]] = tl
		else:
			tl = ml[line[0]]
		tl.append(round(float(line[2]),5))
		ml[line[0]] = tl

	for k in ml.keys():
		print(k, ml[k])


elif a =='4':
	ml = {}
	pth = '/media/data_dump/Ritwik/git/AshaQnA/data/train_loss_'
	linelist = []
	for start, end in mylist:
		for i in range(start, end+1):
			file = open(pth+str(i)+'.txt')
			content = file.readlines()
			file.close()
			linelist += content
	linelist += ['exit']
	for line in linelist:
		if line == 'exit':
			break
		line = line.split()
		if line[0] not in ml.keys():
			tl = []
			ml[line[0]] = tl
		else:
			tl = ml[line[0]]
		tl.append(round(float(line[2]),5))
		ml[line[0]] = tl

	for k in ml.keys():
		print(k, round(st.mean(ml[k]),5), '±', round(st.stdev(ml[k]),5))

	ml = {}
	b = input('Any specific experiment runID?')
	file = open(pth+str(b)+'.txt')
	content = file.readlines()
	file.close()
	linelist = content + ['exit']

	for line in linelist:
		if line == 'exit':
			break
		line = line.split()
		if line[0] not in ml.keys():
			tl = []
			ml[line[0]] = tl
		else:
			tl = ml[line[0]]
		tl.append(round(float(line[2]),5))
		ml[line[0]] = tl

	for k in ml.keys():
		print(k, round(st.mean(ml[k]),5))
