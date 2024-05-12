from flask import Flask, render_template, request
from string import punctuation as string_punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
import re
from heapq import nlargest

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def remove_html_tags_and_numberings(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove numberings in the form [1]
    text = re.sub(r'\[\d+\]', '', text)
    
    return text

@app.route('/summarize', methods=['POST'])
def summarize():
    article_text = request.form['article_text']
    
    article_text = remove_html_tags_and_numberings(article_text)
    
    tokens = word_tokenize(article_text) 

    nltk.download("stopwords")
    stop_words = stopwords.words('english')

    punctuation = string_punctuation + '\n'

    word_frequencies = {}
    for word in tokens:    
        if word.lower() not in stop_words:
            if word.lower() not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    sent_token = sent_tokenize(article_text)

    sentence_scores = {}
    for sent in sent_token:
        sentence = sent.split(" ")
        for word in sentence:        
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    select_length = int(len(sent_token)*0.3)

    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)
    
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
