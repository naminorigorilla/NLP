from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,hamming_loss,f1_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split


import pandas as pd


df = pd.read_csv('../data_set.csv')
df.genres = df['genres'].apply(eval)
df.text_stopword = df['text_stopword'].apply(eval)
df.text_clean = df['text_clean'].astype(str)
df.head()
categories = [
    'Drama',
    'Comedy',
    'Thriller',
    'Action',
    'Romance',
    'Adventure',
    'Crime',
    'Science Fiction',
    'Horror',
    'Family',
    'Fantasy',
    'Mystery',
    'Animation',
    'History',
    'Music',
    'War',
    'Documentary',
    'Western',
    'Foreign',
    'TV Movie'
]


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit_transform(df['genres'])

y = multilabel_binarizer.transform(df['genres'])

for idx, genre in enumerate(multilabel_binarizer.classes_):
  df[genre] = y[:,idx]
  
import pickle
vectorizer = pickle.load(open('../word2vec/tfidf.pkl','rb')) 
X = vectorizer.fit_transform(df['text_clean'])
y = df[categories].copy()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)









LR_pipeline = Pipeline([('clf', OneVsRestClassifier(SVC(kernel='linear',probability=True), n_jobs=-1))])

accuracy = 0
f1 = 0
h1 = 0
auc_roc = 0
i = 0    
for category in categories[:]:
    LR_pipeline.fit(X_train,y_train[category])

    prediction = LR_pipeline.predict(X_test)
    accuracy +=  accuracy_score(y_test[category],prediction)
    f1 += f1_score(y_test[category],prediction,average = 'micro')
    h1 += hamming_loss(y_test[category],prediction)
    auc_roc += roc_auc_score(y_test[category],prediction)
print('Test averaged Accuracy is {}'.format(accuracy/len(categories[:])))
print('Test averaged F1 is {}'.format(f1/len(categories[:])))
print('Test averaged Hamming Loss is {}'.format(h1/len(categories[:])))
print('Test averaged AUC-ROC is {}'.format(auc_roc/len(categories[:])))


#モデルの保存
import pickle
model = LR_pipeline.fit(X_train,y_train[category])
pickle.dump(model,open('../LR_model.pkl',"wb"))