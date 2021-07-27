from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

######################################################
# Base Models
######################################################

def base_models(X, y, scoring):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(verbosity=0,use_label_encoder=False)),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

######################################################
# Automated Hyperparameter Optimization
######################################################


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):

    ###### KNN #######
    knn_params = {"n_neighbors": range(2, 50)}

    ###### CART ######
    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    ###### Random Forests ######
    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    ###### XGBoost ######
    xgboost_params = {"learning_rate": [0.1, 0.01],
                      "max_depth": [5, 8],
                      "n_estimators": [100, 200],
                      "colsample_bytree": [0.5, 1]}

    ###### LightGBM ######
    lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                       "n_estimators": [300, 500, 1500],
                       "colsample_bytree": [0.7, 1]}

    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                   ("CART", DecisionTreeClassifier(), cart_params),
                   ("RF", RandomForestClassifier(), rf_params),
                   ('XGBoost', XGBClassifier(verbosity=0, use_label_encoder=False), xgboost_params),
                   ('LightGBM', LGBMClassifier(), lightgbm_params)]

    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


######################################################
# Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf