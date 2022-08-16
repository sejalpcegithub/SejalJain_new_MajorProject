import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier = pickle.load(open('twitter_review.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index1.html")
@app.route('/index2')
def index2():
    return render_template('index2.html')


@app.route('/index3')
def index3():
    return render_template('index3.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/MiniProjects')
def MiniProjects():
    return render_template('MiniProjects.html')

@app.route('/index4',methods=['GET'])
def index4():
    
    name = str(request.args.get('name'))
    exp = str(request.args.get('exp'))
    
    
    import pandas as pd
    dataset= pd.read_csv('Data_tweet.csv')
    
    
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review)

    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 15000)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 3].values
    
    input_data = [exp] 
    input_data = cv.transform(input_data).toarray()
    input_pred = classifier.predict(input_data)
    input_pred = input_pred.astype(int)
    
    if input_pred[0] == 1:
       
       return render_template('index4.html', prediction_text='Predicted sentiment for given tweets is Positive ')
    
    else:
       return render_template('index4.html', prediction_text='Predicted sentiment for given tweets is Negative ')

    return render_template('index4.html', prediction_text=' Predicted sentiment for given tweets is: {}'.format(input_pred))




if __name__ == "__main__":
    app.run(debug=True)

