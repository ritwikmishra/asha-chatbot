import nltk
from nltk.stem import WordNetLemmatizer 
import requests 
import json
import os
import pickle

lemmatizer = WordNetLemmatizer()
IP=''
def get_docker_ip():
    global IP
    os.system('bash ./server/ip.sh')
    with open('server/IPAddress.txt','r') as file:
        IP=file.read().strip("\n").split()[0]
    return IP

def quillpad(word):
    IP = "172.17.0.2"
    PARAMS = {'inString': word,'lang': 'hindi'}
    URL = "http://"+IP+":8090/processWordJSON"
    print(URL)
    try: 
        data = requests.get(url = URL, params = PARAMS).json()
        result = data['twords'][0]['options'][0]
    except:
        result = word.upper()
    return result  

def quillpad_dict(word):
    with open('input/final.json','r') as fjson:
        data = json.load(fjson)
    try:
        result = data[word.lower()]
    except:
        result = word.upper()
    return result

def seprate(sentence,word_list,bag):
    guess_list = sentence.replace(',','').replace('.','').replace('-','').replace('"','').replace("'",'').split()
    english=[]
    hindi= []
    wrd=[]
    display=[]
    for word in guess_list:
        word = lemmatizer.lemmatize(word.lower())
        wrd.append(word)
        if word in set(word_list) and word not in bag:
            english.append(word)
            display.append(word.upper())
        else:
            hindi.append(word)
            display.append(word)
            # display.append(quillpad_dict(word))
            # display.append(quillpad(word))
    return english,hindi,display




def load_data():
    # English dictionary 3lacs
    # https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
    with open ('input/words_alpha.txt') as file:
        data = file.read()
        word_list = data.strip("\n").split()

    # Hindi stopwords
    # https://github.com/stopwords-iso/stopwords-hi/blob/master/stopwords-hi.txt #has 275 words
    with open ('input/hi_stopwords.pkl','rb') as file: #has 300+ words
        data = pickle.load(file)
        bag = set(data)

    # sample sentences
    with open ('input/data.txt') as file:
        data = file.read()
        sentence_list = data.split("\n")

    return word_list,bag,sentence_list



if __name__=='__main__':
    get_docker_ip()
    list_english=[]
    list_hindi= []
    list_display=[]
    word_list,bag,sentence_list = load_data()
    for sentence in sentence_list:
        hindi,english,display = seprate(sentence,word_list,bag)
        list_english.append(''.join([str(word)+", " for word in english]))
        list_hindi.append(''.join([str(word)+", " for word in hindi]))
        list_display.append(''.join([str(word)+", " for word in display]))

    with open ('output/hindi.txt','w') as file:
        file.writelines("%s\n" % sentence for sentence in list_hindi)

    with open ('output/english.txt','w') as file:
        file.writelines("%s\n" % sentence for sentence in list_english)
        
    """ This txt file is for direct quillpad server connection 

        with open ('output/display-server.txt','w') as file:
        file.writelines("%s\n" % sentence for sentence in list_display)
    """
    """ This txt file is for quillpad dictionary
    
        with open ('output/display-dict.txt','w') as file:
        file.writelines("%s\n" % sentence for sentence in list_display)
    """

    with open ('output/display.txt','w') as file:
        file.writelines("%s\n" % sentence for sentence in list_display)
    