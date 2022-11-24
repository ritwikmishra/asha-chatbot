import requests

database = {}
traindb = []



#fetch info
def getdata(content):
    url = "http://localhost:8880/bot_query"
    payload = {'question_text': content}
    print('='*50,'Sending TELEGRAM message to the AshaQnA server')
    response = requests.request("POST", url, data = payload)
    text = response.json()
    return text

#append to database
def appendentry(ph_key,db_value,oques):
    database[ph_key]={"data":db_value,"oques":oques,"num":0,"flag":0}

def deletentry(ph_number):
    if database:
        database.pop(ph_number)

#check presence
def indb(ph_number):
    try:
        database[ph_number]
    except :
        return False 
    #print("f")
    return True



def answered(ph_number):
    entry = database[ph_number]
    flag = entry["flag"]
    qna = entry["data"]
    q_no = entry["num"] -1 
    oques = entry["oques"]  
    if flag == 1:
        traindb.append([oques,qna[q_no]["Question"]])
        deletentry(ph_number)
        return True
    return False 



def replywithQues(ph_number):
    entry = database[ph_number]
    qna = entry["data"]
    q_no = entry["num"]
    if q_no >= len(qna):
        deletentry(ph_number)
        return "Krpiya Vistaar se prashan dobara puche"
    ques = qna[q_no]["Question"]
    entry["num"] = q_no + 1
    footer = ". Kya aap yeh puchna chate the Haan yah Na"
    return ques+footer



def replywithAns(ph_number):
    entry = database[ph_number]
    flag = entry["flag"]
    if flag == 1:
        entry["flag"] = 0
        return replywithQues(ph_number)
    qna = entry["data"]
    q_no = entry["num"] - 1
    ans = qna[q_no]["Answer"]
    entry["flag"] = 1
    footer = ". Kya aap aur sawaal dekna chahte hai Haan yah Na"
    return ans+footer



def fetchresult(content,ph_number):
    if not indb(ph_number):
        qna = {}
        if content.lower() != 'na' or content.lower() != 'haan':
            qna = getdata(content)
        else:
            msg = "Krpiya dobaara sawaal puche"
            # haan/na in 1st question
            return msg
        appendentry(ph_number,qna,content)
        ques = replywithQues(ph_number)
        return ques
    else:
        if content.lower() == 'na':
            if not answered(ph_number):
                ques = replywithQues(ph_number)
                print(ques)
                return ques
        elif content.lower() == 'haan':
            ans = replywithAns(ph_number)
            print(ans)
            return ans
        else:
            msg = "Krpiya dobaara sawaal puche"
            deletentry(ph_number)
            return msg


def reply(msg, phone_no):
    global traindb
    reply = fetchresult(msg,phone_no)
    if traindb:
        with open('train1.txt', 'a') as f:
            for item in traindb:
                f.write("%s,%s\n" %(item[0],item[1]))
        traindb = []

    if reply == None:
        reply = '''
        Hope your question has been answered. 
        To ask new question do /ask your_ques 
        Example: /ask bacha dudh na piye toh kya kre'''

    return str(reply)