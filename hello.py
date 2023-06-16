from flask import Flask, request
import tensorflow as tf
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.utils import pad_sequences
import re

nltk.download('stopwords')

eng_stops = set(stopwords.words("english"))

ps = SnowballStemmer("english")

app = Flask(__name__)

# Load the tokenizer from the file
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('model.h5')


def process_message(text):
    # remove all the special characters
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)

    # convert all letters to lower case
    words = text.lower().split()

    # remove stop words and lemmatize the words
    words = [ps.stem(word) for word in words if not word in eng_stops]

    # join all words back to text
    return (" ".join(words))


@app.route('/classify', methods=['POST'])
def classify_text():
    # Get the input text from the request
    text = request.form['text']

    # Preprocess the text
    cleaned_text = process_message(text)

    # Tokenize the preprocessed text
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the tokenized text to a fixed length
    padded_sequence = pad_sequences(tokenized_text, maxlen=1250)

    # Make predictions using the model
    predictions = model.predict(padded_sequence)

    # Determine the result based on the prediction
    if predictions[0][0] < 0.4:
        result = "Text is hate speech"
    else:
        result = "Text is not hate speech"

    # Return the result as the API response
    return result


@app.route('/')
def check():
    return 'API is working...'


if __name__ == '__main__':
    app.run(port=8000)
