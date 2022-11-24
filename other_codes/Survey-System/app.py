from flask import Flask, render_template, request
import pandas as pd
app = Flask(__name__)
import json
import csv
import time
import sqlite3
import os,shutil
from datetime import datetime

ques = pd.read_csv("final-Ques_rand.csv")
annotators = ['prerna','nitika','mrinalini']
#           purpl     green    yell       blue      green     yell
colors = ['#f9e6ff','#f2ffe6','#ffffe6','#e6ffff','#ebfaeb','#faf6ea','#ffd6cc']

def getques(page):
    quepp = json.loads(ques[ques.Page==page].to_json(orient="records"))[0]
    #print(quepp)
    return quepp



def getreply(userid,page):
    n = int(page)
    if not os.path.isfile("response/"+userid+".csv"):
        old_name = "response/empty.csv"
        new_name = "response/"+userid+".csv"
        shutil.copy(old_name, new_name)
    reply = pd.read_csv("response/"+userid+".csv")
    reply.index +=1
    res = json.loads(reply[reply.Ques.isin(range(n*3-2,n*3+1))].to_json(orient="records")) #json loads takes care of Hindi characters
    return res


def getdata(userid,page, off=0):
    if userid in annotators:
        anslist = []
        quelist = []

        for i in range(3):
            anslist += getreply(userid,page+i) # fetches the answer list of this user userid
            quelist += [getques(page+i)]

        # print('anslist', anslist, len(anslist))
        # print('quelist', quelist,'\n\n\n\n')
        
        result = anslist,quelist
    # print("The returning value is : ",result)
    else:
        idx = page+off
        if not os.path.isfile("response/d"+userid+".csv"):
            old_name = "response/disagreement.csv"
            new_name = "response/d"+userid+".csv"
            shutil.copy(old_name, new_name)
        df = pd.read_csv("response/d"+userid+".csv")
        b = int(df.loc[idx,'Ques'])
        tq = ques[ques['Page']==int((b+2)/3)].rename(columns = {'OPTION_'+str(1+(b-1)%3):'OPTION', 'ANSWER_'+str(1+(b-1)%3):'ANSWER'})
        quedict = json.loads(tq.to_json(orient="records"))[0]
        ansdict = json.loads(df.loc[idx].to_json())
        ansdict['idx'] = idx
        ansdict['c'] = colors[idx%(len(colors[:-1]))]
        result = ansdict, quedict

    return result



def validate(userid,pas):
    # global user
    # try:    
    #     query = user[user.UniqId==userid].values[0]
    #     password, page = query[1], query[2]
    # except:
    #     return (False,({},{},-1))

    user = pd.read_csv("users/"+userid+".csv")
    password = user.iloc[0,1]
    page = user.iloc[0,2]
     
    try:
        if str(pas) == str(password):
            pdata = getdata(userid,int(page), off=-1)
            result = (True,pdata)
            # print("The returning value is (validate): ",result)
            return result
        else:
            print("exc")
            # (flag,(anslist,quelist,page))
            return (False,({},{}))
    except Exception as e:
        print(e)
        return (False,({},{}))

def writeDisagreements(userid):
    nitika = pd.read_csv('response/nitika.csv')
    prerna = pd.read_csv('response/prerna.csv')

    s1 = set([1,2,3])
    if not os.path.isfile("response/disagreement.csv"):
        df = pd.DataFrame([],columns=list(nitika.columns)+['fAns'])
        df.to_csv('response/disagreement.csv', index=None)

    df = pd.read_csv('response/disagreement.csv')
    d = []
    j = len(df)
    for i in range(len(nitika)):
        if (nitika.loc[i, 'Ans'] != prerna.loc[i,'Ans'] and prerna.loc[i,'Ans'] != 0 and nitika.loc[i,'Ans'] != 0 and nitika.loc[i,'Ques'] not in list(df['Ques'])):
            df.at[j,'Ques'] = nitika.loc[i,'Ques']
            nt = list(s1 - set([nitika.loc[i, 'Ans'],prerna.loc[i,'Ans']]))[0]
            df.at[j,'Ans'] = -nt
            j+=1
    df = df.astype(str)
    df.to_csv('response/disagreement.csv', index=None)

    if not os.path.isfile("response/d"+userid+".csv"):
        old_name = "response/disagreement.csv"
        new_name = "response/d"+userid+".csv"
        shutil.copy(old_name, new_name)

    df2 = pd.read_csv("response/d"+userid+".csv")

    if len(df) > len(df2):
        print('Before', len(df2), end=' | ')
        df2 = df2.append(df.iloc[len(df2):])
        df2 = df2.astype(str)
        df2.to_csv("response/d"+userid+".csv", index=None)
        print('Updated user d response sheet. Now', len(df2))


@app.route('/login',methods = ['POST','GET'])
def login():
    if request.method == 'POST':
        userid = request.form.get("unq_id").lower()
        writeDisagreements(userid)
        pas = request.form.get("unq_pass")
        (flag,(anslist,quelist))  = validate(userid,pas)
        db = {"ques":quelist,"ans":anslist}
        print("The flag is :",format(flag))
        msg = 'Welcome!'
        if flag:
            writeDisagreements(userid)
            if userid in annotators:
                return render_template('index.html',data= {"userid":userid,"db":db, "msg":msg})
            else:
                return render_template('lhmc.html', data={"userid":userid, "db":db, "msg":msg})
        else:
            return render_template('login.html',message={"banner":"Wrong userid or password "+userid})
    return render_template('login.html',message={"banner":""})



def updatepage(userid,page): # save current page
    if userid in annotators:
        user = pd.read_csv("users/"+userid+".csv")
        try:
            user.at[0,'Page'] = page
        except:
            print("userid NOT FOUND IN UPDATEPAGE FUNCTION")
        user.to_csv("users/"+userid+".csv", index=None)
    else:
        user = pd.read_csv("users/"+userid+".csv")
        user.at[0,'Page'] = str(int(page)+1)
        user = user.astype(str)
        user.to_csv("users/"+userid+".csv", index=None)

    


def save(data,userid,page):
    if userid in annotators:
        reply = pd.read_csv("response/"+userid+".csv")
        if 'time' not in reply.columns:
            reply['time']=''
        reply = reply.astype(str)
        reply.index +=1

        page = int(page)
        i = page*3 - 8 # page= 3 then i:1 to 9
        for x in data:
            if not x:
                reply.at[i,"Ans"] = 0
            else:
                print("Saved {} value at index {}".format(x[0],i))
                reply.at[i,"Ans"] = x[0]
                reply.at[i,'time'] = str(datetime.now())
            i+=1

        updatepage(userid,str(int(page)-2))
        reply = reply.astype(str)
        reply.to_csv("response/"+userid+".csv",index=False)
    else:
        reply = pd.read_csv("response/d"+userid+".csv")
        if 'time' not in reply.columns:
            reply['time']=''
        reply = reply.astype(str)
        if data[0]:
            reply.at[int(page),'fAns'] = data[0][0]
            reply.at[int(page),'time'] = str(datetime.now())
        reply = reply.astype(str)
        reply.to_csv("response/d"+userid+".csv",index=False)
        updatepage(userid,page)
    

def fetchdata(request, page, userid):
    data = []
    if userid in annotators:
        n = int(page)
        for x in range(n*3 -8, n*3+1): # 3 becomes 1 to 9
            data.append(request.form.getlist(str(x))) # [1,2,3] this fetches the response
    else:
        data = [request.form.getlist("LHMCanswer")]
    return data

def jumpingpages(userid, page, successchange, failurechange, d):
    page = int(page)
    if userid in annotators:
        if (d=='b' and page>3) or (d=='f' and page<1243):
            (anslist,quelist)  = getdata(userid,page+successchange)
            db = {"ques":quelist,"ans":anslist}
            return True, db
        else:
            (anslist,quelist)  = getdata(userid,page+failurechange)
            db = {"ques":quelist,"ans":anslist}
            return False, db
    else:

        if (d=='b' and page > 0) or (d=='f' and page < len(pd.read_csv("response/d"+userid+".csv"))-1):
            (anslist,quelist) = getdata(userid,max(page+(-1 if successchange == -5 else 1),0))
            db = {"ques":quelist,"ans":anslist}
            return True, db
        else:
            (anslist,quelist) = getdata(userid,page)
            db = {"ques":quelist,"ans":anslist}
            return False, db

            




@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        if request.form.get("prev"):
            data = []
            msg = ''
            page = request.form.get("page")
            userid = request.form.get("userid")

            data = fetchdata(request, page, userid)

            print("\n\n\n\ndata: ",data, datetime.now())
            
            print("page:",page)
            save(data,userid,page)

            t, db = jumpingpages(userid, page, successchange=-5, failurechange=-2, d='b')
            if not t:
                msg = 'ERROR: You cannot go back further'
            if userid in annotators:
                return render_template('index.html',data= {"userid":userid,"db":db,"msg":msg})
            else:
                return render_template('lhmc.html',data= {"userid":userid,"db":db,"msg":msg})

        elif request.form.get("next"):
            data = []
            errList = []
            msg = ''
            page = request.form.get("page")
            userid = request.form.get("userid")

            data = fetchdata(request, page, userid)
            
            print("\n\n\n\ndata: ",data, datetime.now())
            
            save(data,userid,page)

            for i,x in enumerate(data):
                if not len(x):
                    msg = 'ERROR: Please select atleast one option from the given field'
                    errList.append(i+1)
            if msg:
                if userid in annotators:
                    page = str(int(page)-3)
                else:
                    page = str(int(page)-1)


            t, db = jumpingpages(userid, page, successchange=1, failurechange=-2, d='f')
            if msg and userid not in annotators:
                db['ans']['c'] = colors[-1]

            if not t:
                msg = 'ERROR: You cannot go ahead further than this. Contact Ritwik.' if not (msg) else msg+'. And you cannot go ahead further than this.'
            
            print('err', errList)
            if userid in annotators:
                return render_template('index.html',data= {"userid":userid,"db":db, "msg":msg, "err":errList})
            else:
                return render_template('lhmc.html',data= {"userid":userid,"db":db, "msg":msg, "err":errList})

        elif request.form.get("submit"):
            data = []
            page = request.form.get("page")
            userid = request.form.get("userid")
            data = fetchdata(request, page, userid)

            save(data,userid,page)
            return render_template('thankyou.html')
        #print(request.form.getlist("1"))
        
    else:
        return render_template('login.html',message = {"banner":""})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8880,debug = True) 





'''
import pandas as pd
import numpy as np

df = pd.read_csv('final-Ques.csv')

df2 = df.sample(frac=1, random_state=123).reset_index(drop=True)
df2['Page'] = np.arange(1,1009)

df2.to_csv('final-Ques_rand.csv', index=None)

'''


'''
sudo apt install sqlite3
sqlite3 -column -header
.open "admin"

select * from user;
update user set BasePage = 3 where UniqId = 'Doctor';
insert into user values ('Shells','123',10);

'''
