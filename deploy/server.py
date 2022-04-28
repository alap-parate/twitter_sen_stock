from re import template
from tkinter.tix import Form
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sympy import re
from urllib.request import Request
from matplotlib.pyplot import text
from numpy import vectorize
from fastapi import FastAPI,Request
import re
import pickle
from fastapi.templating import Jinja2Templates

class Features(BaseModel):
    tweet_text: str

app = FastAPI(
    title = "Twitter Sentiment Model API",
    description = "Simple API that uses classification to predict the sentiment of tweets",
    version = "0.69"
)

templates = Jinja2Templates(directory='templates')

@app.get("/home", response_class=HTMLResponse)
def home(request: Request, tweet_text: str = Form()):
    tweet = tweet_text
    print(tweet)
    return templates.TemplateResponse("index.html",{"request": request})

model = pickle.load(open("C:\\Users\\Alap Parate\\Desktop\\twitter_sen_stock\\model_xgb.pkl","rb"))
vectorizer = pickle.load(open("C:\\Users\\Alap Parate\\Desktop\\twitter_sen_stock\\transform.pkl","rb"))

def text_cleaning(text):
    text = text.lower()
    hash_pattern  = re.findall("[#|@]\w+",text)
    for p in hash_pattern:
        text = text.replace(p,"")
    urls = re.findall("http[A-Za-z:/0-9.]+",text)
    for u in urls:
        text = text.replace(u,"")
    text =re.findall("\w+",text)
    text = " ".join(text)
    return text  

@app.get("/predict")
def predict_sentiment(tweet:str):
    cleaned_text = text_cleaning(tweet)
    # prediction = model.predict(vectorizer.transform([cleaned_text]))
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    sentiments = {0:'neutral', 1:'positive', -1:'negative'}

    result = {"prediction":sentiments[prediction]}
    # return result
    return result