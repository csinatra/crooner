import keras, pickle, requests, json, time, requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup

with open('../models/cLSTM315-1015/1549314666_LSTM350_1015tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = keras.models.load_model('../models/cLSTM315-1015/1549314666_LSTM315_1015model.h5')

class Generator():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_lyrics(self,
                        seed,
                        seq_len = 4,
                        song_len = 250,
                        temperature = 1.2):
        doc = []
        seed_clean = seed.lower().split(' ')
        for i in seed_clean:
            doc.append(i)

        while len(doc) < (song_len - len(seed_clean)):
            text = [seed_clean]
            sequence = [self.tokenizer.texts_to_sequences([word])[0] for word in text]
            pad_sequence = pad_sequences(sequence, maxlen=seq_len, truncating='pre')
            sequence_reshape = np.reshape(pad_sequence, (1, seq_len))

            yhat = self.model.predict(sequence_reshape, verbose=0)[0]

            preds = np.asarray(yhat).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)

            next_index = np.argmax(probas)

            for word, index in self.tokenizer.word_index.items():
                if index == next_index:
                    seed_clean.append(word)
                    doc.append(word)

        self.output = ' '.join(doc)
        print(self.output)
        return self.output

if __name__ == '__main__':
    active_session = True
    seed = None
    while active_session:
        if not seed:
            print('\n')
            print('Welcome to Crooner')
            print('\n')
            seed = input("Please enter some love lyrics: ")
            gen = Generator(model, tokenizer)
            song = gen.generate_lyrics(seed)
            print('\n\n')

        save_lyrics = input("Would you like to save your lyrics? Y/N")
        if save_lyrics.lower() == 'y':
            name = input("Please enter your name:")
            email = input("Please enter your email address:")
            email = email.lower()
            song += f'\n\n name: {name} \n email: {email} \n seed: {seed}'

            now = round(time.time())
            name = name.lower().replace(' ', '_')
            with open(f'songs/{name}_{now}.txt', "w") as f:
                print(song, file=f)

            more_lyrics = input("Would you like to use the same seed to generate new lyrics? Y/N: ")

            if more_lyrics.lower() == 'y':
                gen.generate_lyrics(seed)
                print('\n\n')
                active_session

            elif more_lyrics.lower() == 'n':
                seed = None
                new_lyrics = input("Would you like to enter a new seed? Y/N :")
                if new_lyrics.lower() == 'y':
                    active_session

                elif new_lyrics.lower() =='n':
                    active_session = False

        elif save_lyrics.lower() == 'n':

            more_lyrics = input("Would you like to use the same seed to generate new lyrics? Y/N: ")
            if more_lyrics.lower() == 'y':
                gen.generate_lyrics(seed)
                print('\n\n')
                active_session

            elif more_lyrics.lower() == 'n':
                seed = None
                new_lyrics = input("Would you like to enter a new seed? Y/N :")
                if new_lyrics.lower() == 'y':
                    active_session

                elif new_lyrics.lower() =='n':
                    active_session = False
