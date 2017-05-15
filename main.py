import numpy as np
import time
import sys
import os
import random
from rnn import *
from sklearn.metrics import accuracy_score
from glob import glob
from sklearn.cross_validation import KFold

def contextwin(l, win):

    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def train_and_eval(user_id):
    
    print "Loading data ... "

    window = 2

    #tr: trajectories
    #tm: timestamps
    #centroids: interest points center

    print "Training...>"
    
    contexts_targets = contextwin(tr, window)
    contexts_targets_t = contextwin(tm, window)
    contexts = []
    targets = []
    for i in range(len(contexts_targets) - 1):
        ct = contexts_targets[i]
        ct2 = contexts_targets_t[i]
        if ct[0] == -1:
            continue
        contexts.append([ct, ct2])
        targets.append(contexts_targets[i + 1][-1])
    
    vlen = len(centroids)
    lr = 0.0627142536696559
    fold = 3
    n_epochs = 100

    data_size = len(contexts)
    if data_size <= fold:
        return None

    k_fold = KFold(data_size, fold)
    contexts = np.asarray(contexts, dtype='int32')
    targets = np.asarray(targets, dtype='int32')
    
    results1 = []
    results2 = []
    results3 = []
    
    for k, (train, test) in enumerate(k_fold):
        best_val1 = 0
        best_val2 = 0
        best_val3 = 0
        
        nn = RNNPredictor(vlen, window=window)
        context_train, targets_train = contexts[train], targets[train]
        context_test, targets_test = contexts[test], targets[test]

        train_size = context_train.shape[0]
        for e in range(n_epochs):
            tic = time.time()
            err = 0.0
            for i, context in enumerate(context_train):
                err += nn.train(context[0], context[1], targets_train[i])

            train_perf = 1 - err / train_size
            
            at_1_correct = 0.0
            at_2_correct = 0.0
            at_3_correct = 0.0

            i=0
            for dev_c, dev_t in zip(context_test, targets_test):
                py = nn.predict(np.asarray(dev_c[0]).astype('int32'), np.asarray(dev_c[1]).astype('int32'))
                if targets_test[i] == py[0]:
                    at_1_correct += 1.0
                    at_2_correct += 1.0
                    at_3_correct += 1.0
                    
                elif targets_test[i] == py[1]:
                    at_2_correct += 1.0
                    at_3_correct += 1.0                

                elif targets_test[i] == py[2] :
                    at_3_correct += 1.0
                    
                i +=1

            acc_at_1 = at_1_correct / len(targets_test)
            acc_at_2 = at_2_correct / len(targets_test)
            acc_at_3 = at_3_correct / len(targets_test)

            print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./train_size), \
                'completed in %.2f (sec) '%(time.time()-tic), \
                'Performance: %f <<' % train_perf, \
                '[validating] >> Performance : @1 => %f \t @2 => %f \t @3 => %f <<\r' % (acc_at_1,acc_at_2,acc_at_3)

            if acc_at_1 > best_val1:
                best_val1 = acc_at_1

            if acc_at_2 > best_val2:
                best_val2 = acc_at_2

            if acc_at_3 > best_val3:
                best_val3 = acc_at_3
                
        print '[cv - %d]  >> Performance : %f \t %f \t %f <<\r' % (k + 1, best_val1, best_val2, best_val3)
        results1.append(best_val1)
        results2.append(best_val2)
        results3.append(best_val3)

    if len(results1) == 3 and len(results1) == 3 and len(results1) == 3:
        accuracy1 = np.mean(results1)
        accuracy2 = np.mean(results2)
        accuracy3 = np.mean(results3)

        print "Accuracy: %f" % accuracy1
        print "Accuracy: %f" % accuracy2
        print "Accuracy: %f" % accuracy3
        
        #writing the results
