
import urllib.request
import urllib
import urllib.parse
from bs4 import BeautifulSoup
from urllib.parse import quote
import datetime
import os
import time
import sys
import random, string, re

# exit()
def myquote(quote_page):
	url = urllib.parse.urlsplit(quote_page)
	url = list(url)
	url[2] = urllib.parse.quote(url[2])
	url = urllib.parse.urlunsplit(url)
	return url



try:
	if True: #("hindi" in str(sys.argv[1]).lower()):
		file_name = 'inshorts_urls.txt'
		last_article_file = "inshorts_dataset/last_article.hi"
		destination_path = "inshorts_dataset/"
	else:
		file_name = "links.txt"
		last_article_file = "dataset/last_article.en"
		destination_path = "dataset/english/"

except:
	print("\tPlease provide 'Hindi' or 'English' after the command. Exit and run again.")
	input()
	exit()
linkfile = open(file_name,"r")

links = linkfile.readlines()
try:
	file = open(last_article_file,"r")
	last_link = file.read().strip('\n')
	file.close()
except:
	last_link = ''
display_top_n = 5
# file = open(last_article_file,"w")
# file.write(links[0])
# file.close()

if last_link == '':
	i = len(links)-1
else:
	i = 0
	while links[i].strip('\n') != last_link:
		i+=1

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,}
hdr = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',}
k=1
# for quote_page in links:
while i > 0:
	quote_page = links[i]
	i-=1
	quote_page = quote_page.strip()
	if not quote_page:
		continue
	# quote_page = "https://inshorts.com/en/news/robert-downey-jr-was-paid-₹524-crore-for-infinity-war-reports-1556721028619"
	# if quote_page == last_link:
	# 	break
	try:
		page = urllib.request.urlopen(myquote(quote_page))
	except Exception as e:
		# print(e)
		continue
	if "FILE_TERMINATES_HERE" in quote_page:
		break
	# dtstring = str(datetime.datetime.now()).replace('.',':')+".txt"
	# quote_page = quote_page.encode('utf-8')
	rstring = ''.join(random.choices(string.ascii_uppercase, k = 4))
	dtstring = destination_path+re.search(r'\d+$',quote_page)[0]+rstring+".txt" #destination_path+quote_page[-13:]+".txt"
	file = open(dtstring,'w')
	# print(quote_page)
	article_len = 0
	
	try:
		soup = BeautifulSoup(page, 'html.parser')
		urlink = soup.find("a", class_="source")['href']
		request = urllib.request.Request(urlink, None, headers)
		response = urllib.request.urlopen(request)
		soup2 = BeautifulSoup(response, 'html.parser')
		file.write("#originalArticleHeadline" + "\n")

	
		if "hindustantimes.com" in urlink:
			hlist = soup2.find('h1')
			hlist2 = soup2.find('h2')
			file.write(hlist.text+"\n")
			file.write(hlist2.text+"\n")

			file.write("#originalArticleBody" + "\n")
			table = soup2.findAll('div',attrs={"class":"story-details"})
			for x in table:
				for p in x.findAll('p'):
					file.write(p.text+"\n")

		elif "thequint.com" in urlink:
			hlist = soup2.find('h1')
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", itemprop="articleBody").findAll('p')
			for a in article:
				file.write(a.text+"\n")

		elif "newindianexpress.com" in urlink:
			hlist = soup2.find('h1', class_="ArticleHead")
			file.write(hlist.text+"\n")
			# straptxt article_summary
			article = soup2.find("div", class_="article_summary").findAll('p')
			for a in article:
				file.write(a.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", class_="articlestorycontent").findAll('p')
			for a in article:
				file.write(a.text+"\n")

		elif "financialexpress.com" in urlink:
			hlist = soup2.find('h1', class_="post-title")
			file.write(hlist.text+"\n")
			hlist = soup2.find('h2', class_="synopsis")
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.findAll('p')
			for a in article:
				file.write(a.text+"\n")

		elif "dailymail.co.uk" in urlink:
			hlist = soup2.find('h2')
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", itemprop="articleBody").findAll('p', class_="mol-para-with-font")
			for a in article:
				file.write(a.text+"\n")

		elif "pib.nic.in" in urlink:
			hlist = soup2.find('div', style="text-align:center;font-weight: bold;")
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", style="text-align: justify;line-height:1.6;font-size:110%").findAll('p', style="text-align:justify")
			for a in article:
				file.write(a.text+"\n")

		elif "aninews.in" in urlink:
			hlist = soup2.find('h1', itemprop="headline")
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", itemprop="articleBody").findAll('p')
			for a in article:
				file.write(a.text+"\n")

		elif "theprint.in" in urlink:
			hlist = soup2.find('h1')
			file.write(hlist.text+"\n")
			hlist = soup2.find('h2')
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.find("div", class_="td-post-content").findAll('p')
			for a in article:
				file.write(a.text+"\n")

		elif "repository.inshorts.com" in urlink:
			hlist = soup2.find('div', class_="title")
			file.write(hlist.text+"\n")
			# print(hlist)
			# input()
			file.write("#originalArticleBody" + "\n")
			table = soup2.findAll('div',class_="content_paragraph")
			for x in table:
				for p in x.findAll('p'):
					file.write(p.text+"\n")

		elif "twitter.com" in urlink or "youtube.com" in urlink or "iplt20.com" in urlink:
			# print(urlink)
			file.close()
			os.remove(dtstring)
			continue

		else:
			hlist = soup2.find('h1')
			file.write(hlist.text+"\n")
			file.write("#originalArticleBody" + "\n")
			article = soup2.findAll('p')
			for a in article:
				file.write(a.text+"\n")
	except:
		file.close()
		os.remove(dtstring)
		print('  --> Waiting of 1 sec. Press Ctrl+C to exit',dtstring,' '*90,end='\r')
		time.sleep(1)
		continue
	
	file.write("\n")
	file.write("-"*100+"\n")
	# file.write(soup.find("span", class_='author').text.strip())
	file.write("#summaryHeadline\n"+soup.find("span", itemprop="headline").text.strip()+"\n")
	file.write("#summaryBody\n"+soup.find("div", itemprop="articleBody").text.strip()+"\n")
	file.write("#datePublished "+soup.find("span", itemprop="datePublished").text.strip()+" ")
	file.write(soup.find("span", clas="date").text.strip()+"\n")
	file.write(soup.find("span", attrs={'class' : 'short'}).text.strip() + " by " + soup.find("span", attrs={'class' : 'author'}).text.strip() + " from News inShorts\n")
	file.write("#reference_link: "+quote_page+"\n")
	file.write("#original_link: "+urlink)
	file.close()

	print(str(i)+" "+quote_page[29:-14]+"\n    Article and Summary pulled!"+" "+datetime.datetime.now().strftime("%H:%M:%S")+' '*100+'\n    file '+dtstring+' '+" "*100+'\n'+'-'*50+' '*100)
	if (k%display_top_n==0):
		print("\033[A"*(display_top_n*4),end='')

	try:
		file = open(last_article_file,"w")
		file.write(quote_page)
		file.close()
	except:
		file = open(last_article_file,"w")
		file.write(quote_page)
		file.close()

	k+=1
	file = open(dtstring, 'r')
	content = file.readlines()
	file.close()
	# time.sleep(2)
	# input('Enter: ')

print("\n"*7+str(k-1)+" Articles and their summaries pulled!")
