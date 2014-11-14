# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:05:12 2013

@author: Mohamed Aly <mohamed@mohamedaly.info>

These are the experiments included in the ACL 2013 paper:
    Mohamed Aly and Amir Atiya.
    LABR: Large-Scale Arabic Book Reviews Dataset.
"""



from __future__ import print_function
import cPickle as pickle
from labr import LABR
import numpy as np
import os
from qalsadi import analex
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier

PATH = './'
file_name = PATH + "experiments_acl2013.txt"

# data sets
gr = LABR()
datas = [
        dict(name="2-balanced", params=dict(klass="2", balanced="balanced")),
        dict(name="2-unbalanced",
             params=dict(klass="2", balanced="unbalanced")),
        dict(name="5-balanced", params=dict(klass="5", balanced="balanced")),
        dict(name="5-unbalanced",
             params=dict(klass="5", balanced="unbalanced"))]

# tokenizer
an = analex()
tokenizer = an.text_tokenize

# features
features = [
            dict(name="count_ng1",
             feat=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))),
            dict(name="count_ng2",
             feat=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,2))),
            dict(name="count_ng3",
             feat=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,3))),
            dict(name="tfidf_ng1",
             feat=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,1))),
            dict(name="tfidf_ng2",
             feat=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,2))),
            dict(name="tfidf_ng3",
             feat=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,3))),
           ]

# classifiers
classifiers = [
               dict(name="nb", clf=MultinomialNB()),
               dict(name="bnb", clf=BernoulliNB(binarize=0.5)),
               dict(name="svm",
                    clf=LinearSVC(loss='l2', penalty="l2",
                                  dual=False, tol=1e-3)),
#               dict(name="forest30",
#                    clf=RandomForestClassifier(n_estimators=30,
#                                               random_state=123,
#                                               verbose=3))]
              ]

# combinations to repeat
repeat = dict() #data="2-unbalanced", feat="tfidf_ng3")

if os.path.exists(file_name):
    scores = pickle.load(open(file_name))
else:
    scores = list()

# go
try:
    idx = -1
    for data in datas:
        # load the data
        print(60*"-")
        print("Loading data:", data['name'])
        (d_train, y_train, d_test, y_test) = gr.get_train_test(**data['params'])

        for feat in features:
#            print("Computing features:", feat['name'])
#            X_train = feat['feat'].fit_transform(d_train)
#            X_test = feat['feat'].transform(d_test)

            for clf in classifiers:
                idx += 1
                # get score if exists and skip if f1 is not nan
                if idx < len(scores) and scores[idx].has_key('f1') and \
                    not np.isnan(scores[idx]['f1']):

                    # check if required to repeat
                    if not(repeat.has_key('data') and repeat['data'] == \
                      data['name'] and repeat.has_key('feat') and \
                      repeat['feat'] == feat['name']):
                        continue

                print("Computing features:", feat['name'])
                # Compute if not already computed for this data
                if not feat.has_key(data['name']):
                    X_train = feat['feat'].fit_transform(d_train)
                    X_test = feat['feat'].transform(d_test)
                    feat[data['name']] = True

                print("Training: ", clf["name"])
                clf['clf'].fit(X_train, y_train)

                print("Testing")
                pred = clf['clf'].predict(X_test)

                # Weighted average of accuracy and f1
                (acc, tacc, support, f1) = (list(), list(), list(), list())
                for l in np.unique(y_test):
                    support.append(np.sum(y_test == l) / float(y_test.size))

                    tp = float(np.sum(pred[y_test == l] == l))
                    fp = float(np.sum(pred[y_test != l] == l))
                    fn = float(np.sum(pred[y_test == l] != l))
                    print("tp:", tp, " fp:", fp, " fn:", fn)
                    if tp>0:
                        prec = tp / (tp + fp)
                        rec = tp / (tp + fn)
                    else:
                        (prec, rec) = (0, 1)

                    f1.append(2*prec * rec / (prec + rec))
                    acc.append(tp / float(np.sum(y_test == l)))
                    tacc.append(tp)

                # compute total accuracy
                tacc = np.sum(tacc) / y_test.size
                # weighted accuracy
                acc = np.average(acc, weights=support)
                # weighted F1 measure
                f1 = np.average(f1, weights=support)

                print("f1 = %0.3f" % f1)
                print("wacc = %0.3f" % acc)
                print("tacc = %0.3f" % tacc)

                score = dict(data=data['name'],
                             feat=feat['name'],
                             clf=clf['name'],
                             f1=f1,
                             acc=acc,
                             tacc=tacc)

                if len(scores) <= idx:
                    scores.append(score)
                else:
                    scores[idx] = score
    print(60*"=")
    for s in scores:
        print("")
        for k,v in s.iteritems():
            print(k, v)

except:
    print("error")

pickle.dump(scores, open(file_name, 'w'))
