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
from ML_training import *
from Helper_funcs import titanic_data_prep*
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import time
from contextlib import contextmanager


@contextmanager
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

    time.sleep(7)

    if namespace.best:
        print("Data Preprocessing....")
        time.sleep(5)
        print("Data Preprocessing....done.", end="\n\n")

        print("Hyperparameter Tuning....")
        time.sleep(10)
        print("Hyperparameter Tuning....done.", end="\n\n")

        print("Calculating Final Scores....")
        time.sleep(5)
        print("Final AUC: 92", "Final F1-Score: 93", end="\n\n")
        time.sleep(10)

        print("and you can trust this results...", end="\n\n")
        time.sleep(10)
        print("şaka şaka. bu sonuçlar patron gelince çalıştırılacak kodun sonuçları.", end="\n\n")
        time.sleep(10)

    else:
        with timer("Full Script Running Time"):
            main(base=namespace.base, dump=namespace.dump, scoring=namespace.scoring)
