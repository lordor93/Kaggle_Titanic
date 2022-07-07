######################################################
# Automated Machine Learning with Custom Pipeline
######################################################
# Hitter icinde bunun benzerini yap

import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Scripts.ML_training import *
from Scripts.Helper_funcs import titanic_data_prep*
from Scripts.Helper_funcs import *

import time
from contextlib import contextmanager


@contextmanager # decorator
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    print(" ")

######################################################
# Main
######################################################
# decorator (sonuc olarak baska method dönüyor)
def main(base, dump, scoring):

    with timer("Data Preprocessing"):
        df = pd.read_csv(r"C:\Users\emrek\Desktop\VBO BootCamp\DSMLBC\DataSets\titanic.csv")
        df_prep = titanic_data_prep(df)
        y = df_prep["SURVIVED"]
        X = df_prep.drop(["PASSENGERID", "SURVIVED"], axis=1)

    if base: # default True
        with timer("Base Models"):
            base_models(X, y, scoring)

    with timer("Hyperparameter Optimization"):
        best_models = hyperparameter_optimization(X, y, cv=3, scoring=scoring)

    with timer("Voting Classifier"):
        voting_clf = voting_classifier(best_models, X, y)

        if dump:
            print("Voting Classifier Model Saved")
            joblib.dump(voting_clf, "voting_clf.pkl")




if __name__ == "__main__": # Main block (same with main class in other OOP languages

    namespace = get_namespace()
    
    with timer("Full Script Running Time"):
        main(base=namespace.base, dump=namespace.dump, scoring=namespace.scoring)
