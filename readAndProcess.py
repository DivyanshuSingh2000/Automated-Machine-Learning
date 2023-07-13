import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class BackEndWork:

    def __init__(self, f_name: str, f_type: str):
        self.f_name = f_name.lower()  # file name only taking
        self.f_type = f_type.lower()
        self.df = pd.DataFrame()

    def reading(self):
        # else one more for error
        path = "uploaded_Files/{}".format(self.f_name)
        if self.f_type == 'json':
            self.df = pd.read_json(path)
        elif self.f_type == 'csv':
            self.df = pd.read_csv(path)
        else:
            self.df = pd.read_excel(path)
        return self.df

    def separating_df(self):
        """
        it will separate df into two part
        1) df with numeric values
        2) df with non-numeric values

        :return: df_numeric,df_obj
        """
        obj_col = list(self.df.dtypes[self.df.dtypes == 'object'].index)  # taking object type value
        cat_col = list(self.df.dtypes[self.df.dtypes == 'category'].index)  # taking category type value
        bool_col = list(self.df.dtypes[self.df.dtypes == 'bool'].index)  # taking bool type value

        # combining categorical, bool and object type in a list
        obj_col.extend(cat_col)
        obj_col.extend(bool_col)
        # cols with numeric values
        numeric_col = list(set(self.df.columns) - set(obj_col))

        # separating dataframe with its values
        df_numeric = self.df[numeric_col]
        df_obj = self.df[obj_col]
        return df_numeric, df_obj

    def preprocessing_df(self, df_numeric, df_obj):

        # working on nan values in both numeric and obj dataframe
        # if na more than 40% then drop the feature
        def drop_feature(col):
            total_values = self.df.count().max()
            col_count = self.df[col].isnull().sum()
            if (col_count / total_values) * 100 > 40:
                return True
            return False

        def missing_cat_values(dff):
            for col in dff.columns:
                if drop_feature(col):
                    dff.drop([col], axis=1, inplace=True)
                else:
                    dff[col].fillna(dff[col].mode()[0], inplace=True)
            return dff

        def missing_numeric_values(dff):
            for col in dff.columns:
                if drop_feature(col):
                    dff.drop([col], axis=1, inplace=True)
                else:
                    impute = KNNImputer(n_neighbors=3)
                    dff[col] = impute.fit_transform(dff[[col]])
            return dff

        df_obj_new = missing_cat_values(df_obj.copy())
        df_numeric_new = missing_numeric_values(df_numeric.copy())
        df = pd.concat([df_numeric_new, df_obj_new], axis=1)

        # performing label encoding on categorical features
        # from sklearn.preprocessing import LabelEncoder
        def encoding_dataframe():
            for feature in df.columns:
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
        encoding_dataframe()

        return df
