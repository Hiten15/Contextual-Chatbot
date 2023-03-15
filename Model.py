import pickle
# restoring all the data structures
data = pickle.load( open( "C:/#MY_FOLDER/ML & DL/Chatbot/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

from tensorflow import keras

model = keras.models.load_model("C:/#MY_FOLDER/ML & DL/Chatbot/chatbot_model.h5")

import json
with open('C:/#MY_FOLDER/ML & DL/Chatbot/intents.json') as json_data:
    intents = json.load(json_data)
    
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import random
import streamlit as st

def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):       #Enumerate() method adds a counter to an iterable and returns it in a form of enumerating object. This enumerated object can then be used directly for loops or converted into a list of tuples using the list() method
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


ERROR_THRESHOLD = 0.30
def classify(sentence):
    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return random.choice(i['responses'])

            results.pop(0)


 
def main():
    
    #giving a title
    st.title('Finance Chatbot')
    
    #getting the input data from user
    
    input = st.text_input('Enter your query')
   
    #code for prediction
    output = ''
    
    #creating a button for prediction
    
    if st.button('Output'):
        output = response(input)
        
    st.success(output)


if __name__ == '__main__':
    main()