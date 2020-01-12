#Libraries for feature encoding
from sklearn.preprocessing import LabelEncoder

#Libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier: Falls under wrapper methods (feature importance)
from sklearn.ensemble import ExtraTreesClassifier #ExtraTreesClassifier: Falls under wrapper methods (feature importance)

#Libraries to measure the accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score


#This function is used to perform classification using DecisionTreeClassifier
def classifyUsingDecisionTreeClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using DecisionTreeClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = DecisionTreeClassifier()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using DecisionTreeClassifier *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier
	
#This function is used to perform classification using LogisticRegression
def classifyUsingLogisticRegression(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using LogisticRegression *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = LogisticRegression()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using LogisticRegression *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier
	
#This function is used to perform classification using LinearDiscriminantAnalysis
def classifyUsingLinearDiscriminantAnalysis(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using LinearDiscriminantAnalysis *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using LinearDiscriminantAnalysis *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier
	
#This function is used to perform classification using GuassianNaiveBayes
def classifyUsingGaussianNB(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using GuassianNaiveBayes *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = GaussianNB()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using GuassianNaiveBayes *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier

#This function is used to perform classification using RandomForestClassifier
def classifyUsingRandomForestClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using RandomForestClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = RandomForestClassifier(n_estimators=700)
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using RandomForestClassifier *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier

#This function is used to perform classification using RandomForestClassifier
def classifyUsingExtraTreesClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using ExtraTreesClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = ExtraTreesClassifier(n_estimators=700)
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using ExtraTreesClassifier *****\n")
    return metrics.accuracy_score(ytest, ytestpred), classifier