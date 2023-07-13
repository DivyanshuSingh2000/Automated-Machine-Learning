from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import pandas as pd


class TrainAndEvaluate:

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.x = df.drop(self.target, axis=1)
        self.y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=.3, random_state=42)

    def features_elimination(self):
        feature_perc = int(len(self.df.columns) * 0.7)  # percentage of features to be selected
        rfecv = RFECV(
            estimator=RandomForestClassifier(),
            min_features_to_select=feature_perc,
            step=5,
            n_jobs=-1,
            scoring="r2",
            cv=5,
        )
        rfecv.fit(self.X_train, self.y_train)
        return self.X_train.columns[rfecv.support_]

    def model_tune(self):

        def random_forest(x, y):  # done
            rf = RandomForestRegressor()
            rf.fit(x, y)
            return rf

        def linear_regression(x, y):  # done
            lr = LinearRegression()
            lr.fit(x, y)
            return lr

        # def naive_bayes(x, y): # not done
        #     nb = GaussianNB()
        #     nb.fit(x, y)
        #     return nb

        def support_vector_regressor(x, y):  # done
            svr = SVR(kernel="linear")
            svr.fit(x, y)
            return svr

        def decision_tree_regressor(x, y):  # done
            dt = DecisionTreeRegressor()
            dt.fit(x, y)
            return dt

        # def xg_boost(x, y):  # not done
        #     xg = XGBClassifier()
        #     xg.fit(x, y)
        #     return xg

        # def light_classifier(x, y):  # not done
        #     lg = LGBMClassifier()
        #     lg.fit(x, y)
        #     return lg

        self.final_features = self.features_elimination()

        random_forest_model = random_forest(self.X_train[self.final_features], self.y_train)
        linear_regression_model = linear_regression(self.X_train[self.final_features], self.y_train)
        # naive_bayes_model = naive_bayes(self.X_train[self.final_features], self.y_train)
        support_vector_regressor_model = support_vector_regressor(self.X_train[self.final_features], self.y_train)
        decision_tree_regressor_model = decision_tree_regressor(self.X_train[self.final_features], self.y_train)
        # xg_boost_model = xg_boost(self.X_train[self.final_features], self.y_train)
        # light_model = light_classifier(self.X_train[self.final_features], self.y_train)

        models = [random_forest_model, linear_regression_model, support_vector_regressor_model,
                  decision_tree_regressor_model]  # , naive_bayes_model, xg_boost_model, light_model
        return models

    @staticmethod
    def accuracy_checker(model, x, y):
        # write the code to check the accuracy in general to all the model
        y_pred = model.predict(x)
        sc = r2_score(y, y_pred)
        return sc

    def best_model(self, models):
        dict = {}  # dict to hold all models accuracy
        for model in models:
            s = str(model)
            dict[s[:s.find('(')]] = self.accuracy_checker(model, self.X_test[self.final_features], self.y_test)
        return dict
