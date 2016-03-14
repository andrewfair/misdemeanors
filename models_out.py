

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
import scipy
import matplotlib.pyplot as plt
import math

# get_ipython().magic('matplotlib inline')



def data(path):
    df = pd.read_csv(path)
    inc = ['identity', 'NUMBER_OF_PERSONS_INJURED', 'NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_INJURED', 
           'NUMBER_OF_PEDESTRIANS_KILLED', 'NUMBER_OF_CYCLIST_INJURED', 'NUMBER_OF_CYCLIST_KILLED', 
           "NUMBER_OF_MOTORIST_INJURED", "NUMBER_OF_MOTORIST_KILLED"]
    new = df[inc]
    new2 = new.groupby('identity').sum()
    new2.columns = ['%s_sum' % i for i in new2.columns]
    new2['inc_sum'] = new2.sum(axis=1)
    new2 = new2.reset_index()
    df2 = df.merge(new2, how='left', left_on='identity', right_on='identity')
    final = ['intersectionID','label','Street_Condition','Traffic_Signal_Condition','Visibility',
         'WetBulbFarenheit','WindSpeed','Precip','PrecipSum','total_involved','bikeLane',
         'NUMBER_OF_PERSONS_INJURED_sum','NUMBER_OF_PERSONS_KILLED_sum','NUMBER_OF_PEDESTRIANS_INJURED_sum',
         'NUMBER_OF_PEDESTRIANS_KILLED_sum','NUMBER_OF_CYCLIST_INJURED_sum','NUMBER_OF_CYCLIST_KILLED_sum',
         'NUMBER_OF_MOTORIST_INJURED_sum','NUMBER_OF_MOTORIST_KILLED_sum','inc_sum']
    # remove columns
    drp = ['Unnamed: 0', 'identity']
    temp = df2.drop(drp, axis=1).dropna(how='any')
    df3 = temp[final]
    # create test validation set
    other, test = train_test_split(df3,  test_size=0.20, random_state=83, stratify=df3.label)
    train, val = train_test_split(other,  test_size=0.30, random_state=83, stratify=other.label)
    # intersection
    val = val.set_index('intersectionID')
    train = train.set_index('intersectionID')
    test = test.set_index('intersectionID')
    
    return train, val, test, df



def BestModel(train, val, p):
    traX = train.values[:,1:]
    traY = train.values[:,0]
    valX = val.values[:,1:]
    valY = val.values[:,0]
    C = np.logspace(1e-5,np.log(3), 20)
    keeper = {}
    scores = []
    for idx, c in enumerate(C):
        clf = SGDClassifier(loss='log', penalty=p, alpha=c, random_state=83, n_iter=5)
        model = clf.fit(traX , traY)
        valscore = model.score(valX,valY)
        scores.append(valscore)
        keeper[idx] = c
    best = C[np.argmax(scores)]
    clf = SGDClassifier(loss='log', penalty=p, alpha=best, random_state=83, n_iter=5)
    best_model = clf.fit(traX , traY)
    valscore = best_model.score(valX,valY)

    return best_model, valscore



def BaggingModel(train, val, n=40000, bags=3, norm='l2'):
    final_coef = np.zeros((bags,train.shape[1]-1))
    ttemp = train.reset_index()
    zeros = ttemp[ttemp.label==0].index.values
    ones = ttemp[ttemp.label==1].index.values
    for i in range(bags):
        zero_bag = shuffle(zeros)[:n]
        one_bag = shuffle(ones)[:n]
        indexes = shuffle(np.concatenate((zero_bag, one_bag)))
        bag = train.iloc[indexes]
        model, score = BestModel(bag, val,p=norm)
        final_coef[i] = model.coef_
        # print('bag %d score: %.5f' % (i, score))
    coef = final_coef.mean(axis=0)
    
    return coef



def Evaluate(test, coef):
    testX = test.values[:,1:]
    testY = test.values[:,0]
    inner = np.dot(coef,testX.T)
    prob = scipy.special.expit(inner)
    
    return prob


def main():
    train, val, test, df = data('../data/vzwV3.csv')
    coef = BaggingModel(train, val, n=15000, bags=10, norm='l1')
    prob = Evaluate(test, coef)
    predict = prob.round()
    test_score = accuracy_score(test.label, predict)
    test_recall = recall_score(test.label, predict)
    test_precision = precision_score(test.label, predict)
    print('accuracy: %.3f, precision: %.3f, recall: %.3f' % (test_score, test_precision, test_recall))
    
    return coef, test.columns



if __name__ == '__main__': 
    coef, columns = main()




