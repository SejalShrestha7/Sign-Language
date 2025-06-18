from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import string
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

def find_video(word):
    path = f"{os.getcwd()}/static/assets/ASL/{word}.mp4"
    print(path)
    return os.path.isfile(path)


def analyze_text(sentence):
    # Tokenizing the sentence
    words = word_tokenize(sentence.lower())

    # Using NLTK's Part-of-Speech tagging
    tagged = nltk.pos_tag(words)

    stop_words = ['@', '#', "http", ":", "is", "the", "are", "am", "a", "it", "was", "were", "an", ",", ".", "?", "!", ";", "/"]
  
    lr = WordNetLemmatizer()
    filtered_text = []
    for w, p in tagged:
        if w not in stop_words and w not in string.punctuation:
            if p in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                filtered_text.append(lr.lemmatize(w, pos='v'))
            elif p in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                filtered_text.append(lr.lemmatize(w, pos='a'))
            else:
                filtered_text.append(w)

    return ' '.join(filtered_text)
