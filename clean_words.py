
text = "When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate the city's hidden corruption and question his family's involvement."

import string
_text = text.translate(str.maketrans('', '', string.punctuation))
str_text = str(_text).lower()


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

lem_text = lemmatize_sentence(str_text)


import nltk
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words("english")

stop_str = [item for item in lem_text if item not in stopwords]

clean_str = ' '.join([item for item in stop_str if len(item)>2])
clean_str