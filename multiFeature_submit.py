# Train: Game1; Test: Game2
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import  sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
#df1 = pd.read_csv('Game1SynReplace.csv')
#df1 = pd.read_csv('Game1OutputAssistreplace.csv')
#df2 = pd.read_csv('Game2SynReplace.csv')
df1 = pd.read_csv('teamsconcatenatedGame1.csv')
df2 = pd.read_csv('teamsconcatenatedGame2.csv')
#df1 = pd.read_csv('Game1OutputGitreplace.csv')
#df2 = pd.read_csv('Game2OutputGitreplace.csv')
#df8 = pd.read_csv('Game1OutputAssistreplace.csv')
#df9 = pd.read_csv('Game2OutputAssistreplace.csv')
df4 = pd.read_csv('Mission1.csv')
df5 = pd.read_csv('Mission2.csv')
df3 = pd.read_csv('GitHubDA.csv')
df1 = df1[['Utterance','Consumer complaint narrative','Product','Consumer complaint narrative4']]
df2 = df2[['Utterance','Consumer complaint narrative','Product','Consumer complaint narrative4']]
df4 = df4[['Product','Consumer complaint narrative','Consumer complaint narrative4','Utterance']]
df5 = df5[['Product','Consumer complaint narrative','Consumer complaint narrative4','Utterance']]
#df1 = df1[['Utterance','Product']]
#df2 = df2[['Utterance','Product']]
df3 = df3[['Utterance','Product','GitHubDA','GitHubpol']]
frames = [df1]
df = pd.concat(frames)
frames1 = [df4,df5]
df6 = pd.concat(frames1)
df.rename(columns = {'Utterance':'narrative'}, inplace = True)
df2.rename(columns = {'Utterance':'narrative'}, inplace = True)
########################################
df.rename(columns = {'Consumer complaint narrative':'narrative1'}, inplace = True)
df.rename(columns = {'Consumer complaint narrative4':'narrative2'}, inplace = True)
df2.rename(columns = {'Consumer complaint narrative':'narrative1'}, inplace = True)
df2.rename(columns = {'Consumer complaint narrative4':'narrative2'}, inplace = True)
#######################################
print(df1.head(10))
df3.rename(columns = {'Utterance':'narrative'}, inplace = True)
df3.rename(columns = {'GitHubDA':'narrative1'}, inplace = True)
df3.rename(columns = {'GitHubpol':'narrative2'}, inplace = True)
df.index = range(62)
df['narrative'].apply(lambda x: str(x))
df['narrative'].apply(lambda x: len(x.split(' '))).sum()
########################################
df['narrative1'].apply(lambda x: len(x.split(' '))).sum()
df['narrative2'].apply(lambda x: len(x.split(' '))).sum()
########################################
cnt_pro = df['Product'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Product', fontsize=12)
plt.xticks(rotation=90)
plt.show();
def model_assessment(y_test,predicted_class,name):
    print("----------start----------------------")
    print(name)
    print('confusion matrix')
    print(sklearn.metrics.confusion_matrix(y_test,predicted_class))
    print('accuracy')
    print(accuracy_score(y_test,predicted_class))
    print('f1 score')
    print(f1_score(y_test, predicted_class, average='weighted'))
    plt.matshow(sklearn.metrics.confusion_matrix(y_test, predicted_class), cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    plt.show()
    print("----------end----------------------")
def print_complaint(index):
    example = df[df.index == index][['narrative', 'Product']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Product:', example[1])
print_complaint(12)
print_complaint(13)
from bs4 import BeautifulSoup
def cleanText(text):
    #text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    text = text.replace('..', '.')
    length = len(text);   
    n = 20;  
    temp = 0;  
    chars = int(length/n);  
    return text
df['narrative'] = df['narrative'].apply(cleanText)
########################################################
df['narrative1'] = df['narrative1'].apply(cleanText)
df['narrative2'] = df['narrative2'].apply(cleanText)
########################################################
df2['narrative'] = df2['narrative'].apply(cleanText)
########################################################
df2['narrative1'] = df2['narrative1'].apply(cleanText)
df2['narrative2'] = df2['narrative2'].apply(cleanText)
#################################################################
train=df
test=df2
print("test len")
print(test)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 1:
                continue
            word=word.split(".")
            for i in word:
                tokens.append(i.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative']), tags=[r.Product]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative']), tags=[r.Product]), axis=1)
##############################################
train_tagged1 = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative1']), tags=[r.Product]), axis=1)
test_tagged1 = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative1']), tags=[r.Product]), axis=1)
train_tagged2 = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative2']), tags=[r.Product]), axis=1)
test_tagged2 = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['narrative2']), tags=[r.Product]), axis=1)
##############################################
print('test set')
print(test_tagged)
print(train_tagged.values[30])
import multiprocessing
cores = multiprocessing.cpu_count()
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    #targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors
def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors
for i in range(30):
    ##########################################################
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=100,window=5,  negative=5, min_count=0, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])
    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=5)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    y_train, X_train = vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = vec_for_learning(model_dmm, test_tagged)
    #######################################################
    model_dmm1 = Doc2Vec(dm=1, dm_mean=1, vector_size=100,window=5,  negative=5, min_count=0, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm1.build_vocab([x for x in tqdm(train_tagged1.values)])
    for epoch in range(30):
        model_dmm1.train(utils.shuffle([x for x in tqdm(train_tagged1.values)]), total_examples=len(train_tagged1.values), epochs=5)
        model_dmm1.alpha -= 0.002
        model_dmm1.min_alpha = model_dmm1.alpha
    y_train1, X_train1 = vec_for_learning(model_dmm1, train_tagged1)
    y_test1, X_test1 = vec_for_learning(model_dmm1, test_tagged1)
    model_dmm2 = Doc2Vec(dm=1, dm_mean=1, vector_size=100,window=5,  negative=5, min_count=0, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm2.build_vocab([x for x in tqdm(train_tagged2.values)])
    
    for epoch in range(30):
        model_dmm2.train(utils.shuffle([x for x in tqdm(train_tagged2.values)]), total_examples=len(train_tagged2.values), epochs=5)
        model_dmm2.alpha -= 0.002
        model_dmm2.min_alpha = model_dmm2.alpha
    y_train2, X_train2 = vec_for_learning(model_dmm2, train_tagged2)
    y_test2, X_test2 = vec_for_learning(model_dmm2, test_tagged2)
    #######################################################
    X_train= np.hstack([X_train, X_train1,X_train2])
    X_test=np.hstack([X_test, X_test1,X_test2])
    #################################################################
    clf = LogisticRegressionCV(
        Cs=10, 
        cv=10,
        tol=0.001,
        max_iter=1000,
        scoring="accuracy",
        verbose=False,
        multi_class='ovr',
        random_state=5434
    )
    clf.fit(X_train, y_train)
    LogisticRegressionCV(Cs=10, class_weight=None, cv=10, dual=False,
           fit_intercept=True, intercept_scaling=1.0, max_iter=1000,
           multi_class='ovr', n_jobs=None, penalty='l2', random_state=5434,
           refit=True, scoring='accuracy', solver='lbfgs', tol=0.001,
           verbose=False)
    y_pred = clf.predict(X_test)
    accuracy4=accuracy_score(y_test, y_pred)
    f4=f1_score(y_test, y_pred, average='weighted')
    name="log"
    model_assessment(y_test, y_pred,name)
##############################################
#################################################
    model_svm=SVC()
    model_svm.fit(X_train, y_train)
    y_pred =model_svm.predict(X_test)
    accuracy5=accuracy_score(y_test, y_pred)
    f5=f1_score(y_test, y_pred, average='weighted')
    name="svm"
    model_assessment(y_test, y_pred,name)
#################################################
#-------------------
    
