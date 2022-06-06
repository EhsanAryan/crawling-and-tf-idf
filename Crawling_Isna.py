import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import pandas as pd
from termcolor2 import colored
import pyfiglet as pf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Functions
def repetitious(links , urls):
    for link in links:
        if link.get('href') in urls:
            return True
    return False
    
    
def crawling_function(year , first_page=1 , last_page=100):
    page = first_page-1
    scraped_data = []
    url_list = []
    
    while True :
        page += 1

        page_url = f"https://www.isna.ir/page/archive.xhtml?mn=3&wide=0&dy=4&ms=0&pi={page}&yr={year}"

        page_html = requests.get(page_url).text

        soup = BeautifulSoup(page_html , features="lxml")
        
        links = soup.find_all('a')


        if page>last_page or repetitious(links , url_list):
            break
            

        cnt = 0
        for link in tqdm(links):
            if "http" not in link.get('href') and ("news/" in link.get('href') or "photo/" in link.get('href')) and link.get('href') not in url_list:
                news_url = 'https://isna.ir' + link.get('href')
                url_list.append(link.get('href'))
                cnt += 1
                try:
                    article = Article(news_url)
                    article.download()
                    article.parse()
                    scraped_data.append({'url': news_url, 'title': article.title, 'text': article.text})
                except:
                    print(f"Occurred an error in connection!\nCan't process this page:\n{news_url}")
            if cnt==30:
                break
        
        if cnt<30 and cnt!=0:
            scraped_data.pop()
            url_list.pop()


    if len(scraped_data)==0:
        print(colored(f"There Is No Archive For {year}th Year!\nTry Again!" , color="red"))
        exit()
    else:
        data_frame = pd.DataFrame(scraped_data)
        data_frame.to_csv(f'./Desktop/IR_Project/isna-data-{year}.csv')

    # print(f"The number of Scraped data : {len(scraped_data)}")



# Main
while True:
    try:
        start_page = int(input("Enter the number of start page : "))
        while start_page <= 0 or start_page > 100:
            print(colored("Error! Start Page Must Be A Number From 1 To 100!" , color="red"))
            start_page = int(input("Enter the number of start page again : "))
    except ValueError:
        print(colored("An Value Error Occurred! Start Page Must Be An Integer Number!" , color="red"))
    else:
        break

while True:
    try:
        end_page = int(input("Enter the number of end page : "))
        while end_page <= 0 or end_page > 100:
            print(colored("Error! End Page Must Be A Number From 1 To 100!" , color="red"))
            end_page = int(input("Enter the number of end page again : "))
        while end_page < start_page:
            print(colored("Error! The End Page Must Be Greater Than or equal To The Start Page!" , color="red"))
            end_page = int(input("Enter the number of end page again : "))
    except ValueError:
        print(colored("An Value Error Occurred! End Page Must Be An Integer Number!" , color="red"))
    else:
        break

while True:
    try:
        year = int(input("Enter the year : "))
        while year <= 0:
            print(colored("Error! Year Must Be A Positive Number!" , color="red"))
            year = int(input("Enter the year again : "))
    except ValueError:
        print(colored("An Value Error Occurred! Year Must Be An Integer Number!" , color="red"))
    else:
        break

# Crawling data
crawling_function(year, start_page, end_page)


# Reading file
df = pd.read_csv(f'./Desktop/IR_Project/isna-data-{year}.csv')
pd.options.display.max_rows = 9999

url_list = list(df['url'])
title_list = list(df['title'])
text_list = list(df['text'])


# Applying tf-idf on docs
print("-------------------------- Sections --------------------------")
print("1) Query on text of news.\n2) Query on title of news.")
print("--------------------------------------------------------------")
while True:
    try:
        section = int(input("Enter the section : "))
        while section < 1 or section > 2:
            print(colored("Error! Out of Range!" , color="red"))
            section = input("Enter the section again : ")
    except ValueError:
        print(colored("An Value Error Occurred! Section Must Be An Integer Number!" , color="red"))
    else:
        break

vectorizer = TfidfVectorizer()
if section==1:
    tfidf_docs = vectorizer.fit_transform(df['text'].astype('U').values)
elif section==2:
    tfidf_docs = vectorizer.fit_transform(df['title'].astype('U').values)


query = None
while True:
    query = input(colored("Enter the query (Enter \"q\" to Exit) : " , color="yellow"))
    if query=="q" or query=="Q":
        print(colored(pf.figlet_format("THE END!") , color="green"))
        break
    tfidf_query = vectorizer.transform([query])[0]
    # print(tfidf_query)

    # Cosine Similarity
    cosines = []
    for doc in tqdm(tfidf_docs):
        cosines.append(float(cosine_similarity(doc , tfidf_query)))

    # Sorting
    while True:
        try:
            count = int(input(colored("Enter the number of top documents you want : " , color="yellow")))
            while count < 1 :
                print(colored("Error! Input Must Be An Positive Number!" , color="red"))
                count = int(input(colored("Enter the number of top documents you want again : " , color="yellow")))
        except ValueError:
            print(colored("An Value Error Occurred! Input Must Be An Integer Number!" , color="red"))
        else:
            break
    
    # Show docs
    print(colored("---------------------------------------------------------------------------------------------------------\n" , color="cyan"))
    sorted_ids = np.argsort(cosines)
    for i in range(count):
        cur_id = sorted_ids[-i-1]
        print(f"The {i+1}th document URL :\n{url_list[cur_id]}")
        print(f"The {i+1}th document title :\n{title_list[cur_id]}")
        print(f"The {i+1}th document text:\n{text_list[cur_id]}")
        print(colored(f"\033[1mCosine similarity of {i+1}th document: {cosines[cur_id]}\033[0m" , color="green"))
        print(colored("---------------------------------------------------------------------------------------------------------\n" , color="cyan"))
