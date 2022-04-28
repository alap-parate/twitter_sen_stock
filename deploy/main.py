from fastapi import FastAPI,Request,Form
import re
import pickle
from fastapi.templating import Jinja2Templates

# class Features(BaseModel):
#     tweet_text : str
templates = Jinja2Templates(directory='templates/')

app = FastAPI(
    title = "Twitter Sentiment Model API",
    description = "Simple API that uses classification to predict the sentiment of tweets",
    version = "0.69"
)

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

@app.get('/')
def home():
	return {'message': 'Welcome to the Tweets Sentiments API'}

# @app.get('/form')
# def form_post(request: Request):
#     result = " "
#     return templates.TemplateResponse('index.html',context={'request': request, 'result':result})

@app.post("/form")
def predict_sentiment(request: Request, tweet_box: str = Form(...)):
    result= ""
    cleaned_text = text_cleaning(tweet_box)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    sentiments = {0:'neutral', 1:'positive', -1:'negative'}

    result = {sentiments[prediction]}
    # return result
    # return result
    return templates.TemplateResponse('index.html',context={'request':request, 'result':result ,'tweet':cleaned_text})
    # return sentiments[1.0]


