# 1. Install

1. pip install use the requirements.txt file

   if some libraries gives you error like ```dataclasses``` then install them separately

   You will have to install the indic-trans library by cloning their repo. Instructions [here](https://github.com/libindic/indic-trans).

2. Models

   Download 695_epoch_3.pth.tar file from [here](https://drive.google.com/file/d/1HRTsqrwKYWlKB_d0UzUxlzNo0KaVCDyw/view?usp=sharing) and place the file parallel to ```app.py``` file.  

3. Data
   
   Download allQnA_small.csv file from [here](https://drive.google.com/file/d/1z-oYQ1c_2mgcWGhdySfZ5vh2wlJldCQx/view?usp=share_link) and place the file parallel to ```app.py``` file.  

   The URLs of the Inshorts news articles are given in ```/other_codes/SentSimHi_BERT_spc/data/inshorts_urls.txt```. Code to scrape the articles is given as well.

3. For the first time run, it will download some models first. Hence, might take some time.

# 2. Run the chatbot

For the web interface (flask app) simply run the ```app.py``` file.

## a) Twilio/Whatsapp

Open four terminal windows in the same directory

Run the following in them respectively

1. ```app.py``` flask app

2. ```main_twilio.py``` flask app

3. Run ngrok server

 * Go to [website](https://ngrok.com). You will need to register if you are new. The binary file is already in this repo.  
 * First authenticate (you need to have an account on ngrok) ```./ngrok authtoken <your token here>```.<br>
 * Then run the ngrok server by ```./ngrok http 5123```

Copy the https://abcd123etcetc.ngrok.io URL from ngrok and concat it with /sms to form a URL like https://abcd123etcetc.ngrok.io/sms

Paste this final URL to twilio webhook URL

Watch this video (part1+2) if you get stuch anywhere (for twilio part).

[![](http://img.youtube.com/vi/BKK5NMDC0fk/0.jpg)](http://www.youtube.com/watch?v=BKK5NMDC0fk "")

## b) Telegram

In order to deploy the bot on telegram, open the telegram_bot folder to see the instructions

# 3. Sample Inputs

1. दूध किस पोजीशन में पिलाना चाहिए? (In what position should we feed milk (to an infant)?)

 * Top suggestion by the chatbot is: बच्चे को किस <पोजीशन> में <फीड> करना चाहिए (In what position should we breastfeed an infant?)

2. chota bachcha din me sota hai raat ko nahin sota kya karein? (An infant sleeps during the day but does not sleeps in the night. What to do?)

 * Top suggestion by the chatbot is:  छोटे बच्चे रात को खेलते या सोते है और रात को खेलते या रोते है और दिन में वह बहुत सोते है तो इसलिए हम क्या करे (Infants sleeps a lot during the day but in the night they play or cry. What to do in this situation?)

# 4. Methodology

User (either ASHA worker or a young mother) gives us the query they want to ask. And we search the most similar question to their query from our database of question-answer pairs.  
We have added four ways by which our website can fetch the most similar questions.

1. Uses simple lemmatization and word matching. To select this way, make sure your query starts with the word "Lem". For eg, '_Lem bachche ko kaise rap karein?_'.

2. Uses pruned dependency tree and keyword matching. To select this way, start your query with "DTP". For eg, '_DTP bachche ko kaise rap karein?_'.

3. Uses the BERT model we trained which checks similarity between two hindi sentences. Dataset used was Inshorts, and the accuracy observed was 95%. To select this way, start your query with "BERT_spc". For eg, '_BERT\_spc bachche ko kaise rap karein?_'.

4. Uses BERT sentence embeddings and cosine similarity. This is the default one. But to explicitly select this way, start your query with 'BERT_cos'.

We have kept four ways, so that we can test their effectiveness using user feedback. By default, the chatbot uses an ensemble method of the last three methods.
