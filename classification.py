import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score


class TrainAndEvaluate:

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.x = df.drop(self.target, axis=1)
        self.y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=.3, random_state=42, stratify=self.y)

    def features_elimination(self):
        feature_per = int(len(self.df.columns) * 0.7)  # percentage of features to be selected
        rfecv = RFECV(
            estimator=RandomForestClassifier(),
            min_features_to_select=feature_per,
            step=5,
            n_jobs=-1,
            scoring="r2",
            cv=5,
        )
        rfecv.fit(self.X_train, self.y_train)
        return self.X_train.columns[rfecv.support_]

    def model_tune(self):

        def random_forest(x, y):
            rf = RandomForestClassifier()
            rf.fit(x, y)
            return rf

        def logistic_regression(x, y):
            lg = LogisticRegression(solver='liblinear')
            lg.fit(x, y)
            return lg

        def naive_bayes(x, y):
            nb = GaussianNB()
            nb.fit(x, y)
            return nb

        def support_vector_classifier(x, y):
            svc = SVC(decision_function_shape='ovo')
            svc.fit(x, y)
            return svc

        def decision_tree_classifier(x, y):
            dt = DecisionTreeClassifier()
            dt.fit(x, y)
            return dt

        def xg_boost(x, y):
            xg = XGBClassifier()
            xg.fit(x, y)
            return xg

        def light_classifier(x, y):
            lg = LGBMClassifier()
            lg.fit(x, y)
            return lg

        self.final_features = self.features_elimination()   # calling feature elimination method

        random_forest_model = random_forest(self.X_train[self.final_features], self.y_train)
        logistic_regression_model = logistic_regression(self.X_train[self.final_features], self.y_train)
        naive_bayes_model = naive_bayes(self.X_train[self.final_features], self.y_train)
        support_vector_classifier_model = support_vector_classifier(self.X_train[self.final_features], self.y_train)
        decision_tree_classifier_model = decision_tree_classifier(self.X_train[self.final_features], self.y_train)
        xg_boost_model = xg_boost(self.X_train[self.final_features], self.y_train)
        light_model = light_classifier(self.X_train[self.final_features], self.y_train)

        models = [random_forest_model, logistic_regression_model, naive_bayes_model, support_vector_classifier_model,
                  decision_tree_classifier_model, xg_boost_model, light_model]
        return models

    def accuracy_checker(self, model, x, y):
        y_pred = model.predict(x)
        f1_sc = f1_score(y, y_pred)
        return f1_sc

    def best_model(self, models):
        d = {}  # dict to hold all models accuracy
        for model in models:
            s = str(model)
            d[s[:s.find('(')]] = self.accuracy_checker(model, self.X_test[self.final_features], self.y_test)
        return d



    # include try catch in all the files
