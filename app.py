from flask import Flask, jsonify, request,render_template
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings


nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')
# nlp = load('en_core_web_sm')
nltk.download('punkt')
import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
from googlesearch import search
import webbrowser
import requests
import billboard
import time
from pygame import mixer
import COVID19Py
import urllib.request
import bs4 as bs
from keras.models import load_model

model = load_model('Models/mymodel.h5')
model.summary()
intents = json.loads(open('Models/codeintent.json').read())
words = pickle.load(open('Models/words.pkl','rb'))
classes = pickle.load(open('Models/classes.pkl','rb'))

'''
This method takes the user query as input and returns the response from google search.
It includes condition for "Stackoverflow", if user asks a coding question that goes to the 
StackOverflow website.
'''
def googleSearch(query):
    for link in search(query, tld="co.in", num=10, stop=10, pause=2):
        try:
            raw_html = urllib.request.urlopen(link)
            raw_html = raw_html.read()
            article = bs.BeautifulSoup(raw_html,'lxml')
            if "stackoverflow.com" in link:
                items = article.find_all(class_="answercell post-layout--right")
                for item in items:
                    para = item.findNext('p').text
                    code = item.findNext('code').text
                    break
                return " ".join((para,code))
            else:
                para = article.find_all("p")
                text = ""
                for p in para[:3]:
                    text += p.text
                text = text
                return text
            break
        except Exception as e:
            continue

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def googleSearch(query):
    for link in search(query, tld="co.in", num=10, stop=10, pause=2):
        try:
            raw_html = urllib.request.urlopen(link)
            raw_html = raw_html.read()
            article = bs.BeautifulSoup(raw_html,'lxml')
            if "stackoverflow.com" in link:
                items = article.find_all(class_="answercell post-layout--right")
                for item in items:
                    para = item.findNext('p').text
                    code = item.findNext('code').text
                    break
                return " ".join((para,code))
            else:
                para = article.find_all("p")
                text = ""
                for p in para[:3]:
                    text += p.text
                text = text
                return text
            break
        except Exception as e:
            continue

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

'''
The method responsible to get the response
based on the specific tag predicted.
'''
def get_response(return_list,intents_json):
    
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']
    if tag=='datetime':
        return("Today is : "+time.strftime("%A")+" Date : "+time.strftime("%d %B %Y")+" Time :",time.strftime("%H:%M:%S"))        
        print(time.strftime("%A"))
        print (time.strftime("%d %B %Y"))
        print (time.strftime("%H:%M:%S"))
    if tag=='google':
        query=input('Enter query...')
        res = googleSearch(query)
        print(res)
    if tag=='weather':
        api_key='987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name : ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url) 
        x=response.json()
        print('Present temp.: ',round(x['main']['temp']-273,2),'celcius ')
        print('Feels Like:: ',round(x['main']['feels_like']-273,2),'celcius ')
        print(x['weather'][0]['main'])  
    if tag=='news':
        main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []  
        for ar in article: 
            results.append([ar["title"],ar["url"]])  
        for i in range(3): 
            print(i + 1, results[i][0])
            print(results[i][1],'\n')
    if tag=='song':
        chart=billboard.ChartData('hot-100')
        print('The top 10 songs at the moment are:')
        for i in range(10):
            song=chart[i]
            print(song.title,'- ',song.artist)       
    if tag=='timer':        
        mixer.init()
        x=input('Minutes to timer..')
        time.sleep(float(x)*60)
        mixer.music.load('Handbell-ringing-sound-effect.mp3')
        mixer.music.play()  
#     if tag=='covid19':
#         covid19=COVID19Py.COVID19(data_source='jhu')
#         country=input('Enter Location...')
#         if country.lower()=='world':
#             latest_world=covid19.getLatest()
#             print('Confirmed:',latest_world['confirmed'],' Deaths:',latest_world['deaths'])
#         else:     
#             latest=covid19.getLocations()
#             latest_conf=[]
#             latest_deaths=[]
#             for i in range(len(latest)):
#                 if latest[i]['country'].lower()== country.lower():
#                     latest_conf.append(latest[i]['latest']['confirmed'])
#                     latest_deaths.append(latest[i]['latest']['deaths'])
#             latest_conf=np.array(latest_conf)
#             latest_deaths=np.array(latest_deaths)
#             print('Confirmed: ',np.sum(latest_conf),'Deaths: ',np.sum(latest_deaths))
    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result


def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents)
    return response

def display_result(query,func):
    print("You: ",query)
    print("Chatbot: ", func(query))



app = Flask(__name__,template_folder='template')

# Sample data
books = [
    {
        'id': 1,
        'title': 'Python Programming',
        'author': 'John Doe'
    },
    {
        'id': 2,
        'title': 'Web Development 101',
        'author': 'Jane Smith'
    }
]

# Route for getting all books
@app.route('/api/books', methods=['GET'])
def get_books():
    return jsonify(books)

# Route for getting a specific book
@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        return jsonify(book)
    return jsonify({'error': 'Book not found'}), 404

# Route for creating a new book
@app.route('/api/books', methods=['POST'])
def create_book():
    new_book = {
        'id': len(books) + 1,
        'title': request.json.get('title'),
        'author': request.json.get('author')
    }
    books.append(new_book)
    return jsonify(new_book), 201

@app.route('/api/chat', methods=['GET','POST'])
def get_html_address():
    app.jinja_env.cache.clear()

    return render_template('index.html')

@app.route('/api/ask/<parameter>', methods=['GET','POST'])
def GetResponse(parameter):
    app.jinja_env.cache.clear()

    return response(parameter)

@app.route('/api/msg/local', methods=['POST'])
def generate_response():
    data = request.get_json()
    message = data['message']
    
    # Process the user's message and generate a response
    response2 = response(message)    
    return jsonify({'message': response2})

# Run the app
if __name__ == '__main__':
    app.run()
