import keras, pickle, requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/lyrics', methods=['POST'])

def generate_lyrics(seq_len = 4,
                    song_len = 250,
                    temperature = 1.0):
    
    with open('../models/cLSTM315-1015/*.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    model = keras.models.load_model('../models/cLSTM315-1015/*.h5')

    seed = request.form['seed']
    
    if not seed:
        return redirect(url_for('index'))
    
    seed_clean = seed.lower().split(' ')
    doc = []
    
    while len(doc) < song_len:
        text = [seed_clean]
        sequence = [tokenizer.texts_to_sequences([word])[0] for word in text]
        pad_sequence = pad_sequences(sequence, maxlen=seq_len, truncating='pre')
        sequence_reshape = np.reshape(pad_sequence, (1, seq_len))

        yhat = model.predict(sequence_reshape, verbose=0)[0]
            
        preds = np.asarray(yhat).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        next_index = np.argmax(probas)
        
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                seed_clean.append(word)
                doc.append(word)

    result = ' '.join(doc)
    result = result.split('\n')
    
    return render_template('lyrics.html', result=result, seed=seed)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)