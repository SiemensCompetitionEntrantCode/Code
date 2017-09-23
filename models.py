import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Import morphokinetic data and genetic analysis information
# load datasets
features = np.genfromtxt('morphokinetic_data.csv', dtype = float, delimiter=',')
labels = np.genfromtxt('genetic_reults.csv', dtype = int, delimiter=',')

# Create individual machine learning models and perform
# K-fold cross validation (k=10)
def FNN_k_fold_validation():
    # fix random seed for reproducibility
    seed = 431
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # create list to append model scores
    cvscores = []
    # K-fold validation
    for train_index, test_index in kfold.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        # standardize training data
        scaler = preprocessing.StandardScaler().fit(features_train)
        features_train_scaled = scaler.transform(features_train)
        # create model
        model = Sequential()
        model.add(Dense(30, activation='relu', input_dim=17))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        model.fit(features_train, labels_train, epochs=500, batch_size=15, verbose=0)
        # evaluate the model
        # first scale test data according to scaler object from training data
        features_test_scaled = scaler.transform(features_test)
        scores = model.evaluate(features_test_scaled, labels_test, verbose=0)
        print "acc:", scores[1]*100
        cvscores.append(scores[1]*100)
    return("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def SVM_k_fold_validation():
    # fix random seed for reproducibility
    seed = 164
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # create list to append model scores
    cvscores = []
    # K-fold validation
    for train_index, test_index in kfold.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        # standardize data
        scaler = preprocessing.StandardScaler().fit(features_train)
        features_train_scaled = scaler.transform(features_train)
        # create model
        clf = svm.SVC()
        clf.fit(features_train_scaled, labels_train)
        # evaluate the model
        # first scale test data according to scaler object from training data
        features_test_scaled = scaler.transform(features_test)
        scores = clf.score(features_test_scaled, labels_test)
        #print scores
        print "acc:", scores*100
        cvscores.append(scores*100)
    return("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def RF_k_fold_validation():
    # fix random seed for reproducibility
    seed = 253
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # create list to append model scores
    cvscores = []
    # K-fold validation
    for train_index, test_index in kfold.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        # standardize data
        scaler = preprocessing.StandardScaler().fit(features_train)
        features_train_scaled = scaler.transform(features_train)
        # create model
        clf = RandomForestClassifier(n_estimators=10000)
        clf.fit(features_train_scaled, labels_train)
        # evaluate the model
        # first scale test data according to scaler object from training data
        features_test_scaled = scaler.transform(features_test)
        scores = clf.score(features_test_scaled, labels_test)
        #print scores
        print "acc:", scores*100
        cvscores.append(scores*100)
    return("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Ensemble Model
# Ensemble model averages the probability prediction of the 
# individually trained models
# First create the individual models:

# neural network function
def create_FNN(f_trn, l_trn, f_tst, l_tst):
    # fix random seed for reproducibility
    seed = 43
    np.random.seed(seed)
    # create model
    model = Sequential()
    model.add(Dense(30, activation='relu', input_dim=17))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.fit(f_trn, l_trn, epochs=100, batch_size=20, verbose=0)
    # evaluate the model
    scores = model.evaluate(f_tst, l_tst, verbose=0)
    print "FNN Acc:", scores[1]*100
    return model

# SVM function
def create_SVM(f_trn, l_trn, f_tst, l_tst):
    # fix random seed for reproducibility
    seed = 43
    np.random.seed(seed)
    model = svm.SVC(probability=True)
    model.fit(f_trn, l_trn)
    scores = model.score(f_tst, l_tst)
    print "SVM Acc:", scores*100
    return model

# RF function
def create_RF(f_trn, l_trn, f_tst, l_tst):
    # fix random seed for reproducibility
    seed = 98
    np.random.seed(seed)
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(f_trn, l_trn)
    scores = model.score(f_tst, l_tst)
    print "RF Acc:", scores*100
    return model

# Use individual models to create the ensemble model:
# create ensemble model evaluator function
def ensemble_eval(x, l, net_model, svm_model, rf_model):
    net_pred = net_model.predict(x)
    svm_pred = svm_model.predict_proba(x)
    svm_pred = svm_pred[:,1]
    rf_pred = rf_model.predict_proba(x)
    rf_pred = rf_pred[:,1]
    y_pred = np.array([(x + y + z)/3 for x, y, z in zip(net_pred, svm_pred, rf_pred)])
    y_rounded = np.around(y_pred)
    acc = accuracy_score(l, y_rounded)
    return acc

# Evaluate the ensemble model with k-fold validation:
def ens_k_fold_validation():
    # fix random seed for reproducibility
    seed = 62
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed)
    # create list to append model scores
    cvscores = []
    # K-fold validation
    for train_index, test_index in kfold.split(features, labels):
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        # standardize data
        scaler = preprocessing.StandardScaler().fit(features_train)
        features_train_scaled = scaler.transform(features_train)
        # scale test data according to scaler object from training data
        features_test_scaled = scaler.transform(features_test)
        # create individual models
        net_model = create_ANN(features_train_scaled, labels_train, 
                               features_test_scaled, labels_test)
        svm_model = create_SVM(features_train_scaled, labels_train, 
                               features_test_scaled, labels_test)
        rf_model = create_RF(features_train_scaled, labels_train, 
                               features_test_scaled, labels_test)
        # evaluate the ensemble model based on the models just trained above
        scores = ensemble_eval(features_test_scaled, labels_test, 
                               net_model, svm_model, rf_model)
        # print scores
        print "ENS Acc:", scores*100
        cvscores.append(scores*100)
        print "---------------"
    return("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
