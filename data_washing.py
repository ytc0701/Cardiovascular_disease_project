import pandas as pd

class CardioPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def delete_strange(self):
        df = self.df
        # 年齡限制
        df = df[(df['age'] / 365 >= 0) & (df['age'] / 365 <= 150)]
        # 身高體重
        df = df[(df['height'] >= 100) & (df['height'] <= 300)]
        df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        # 血壓範圍
        df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 200)]
        df = df[(df['ap_lo'] >= 0) & (df['ap_lo'] <= 130)]
        df = df[df['ap_hi'] > df['ap_lo']]
        self.df = df
        return self

    def one_hot_from_map(self, col, mapping, prefix):
        for val, name in mapping.items():
            self.df[f"{prefix}_{name}"] = (self.df[col] == val).astype(int)
        self.df.drop(columns=[col], inplace=True)
        return self

    def categorize_and_onehot(self, col, bins, labels, prefix):
        self.df[f"{col}_group"] = pd.cut(self.df[col], bins=bins, labels=labels, right=True)
        dummies = pd.get_dummies(self.df[f"{col}_group"], prefix=prefix, dtype=int)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[col, f"{col}_group"], inplace=True)
        return self

    def process_age(self):
        age_bins = [28, 34, 44, 54, 64]
        age_labels = ['young_adult','early_middle','middle','late_middle']
        self.df['age_years'] = self.df['age'] // 365
        self.categorize_and_onehot('age_years', age_bins, age_labels, 'age')
        self.df.drop(columns=['age'], inplace=True)
        return self

    def process_gender(self):
        self.df['male'] = (self.df['gender'] == 1).astype(int)
        self.df['female'] = 1 - self.df['male']
        self.df.drop(columns=['gender'], inplace=True)
        return self

    def process_bmi(self):
        bmi_bins = [0, 18.5, 25, 30, 100]
        bmi_labels = ['underweight','normal','overweight','obese']
        self.df['BMI'] = self.df['weight'] / ((self.df['height']/100)**2)
        self.categorize_and_onehot('BMI', bmi_bins, bmi_labels, 'BMI')
        self.df.drop(columns=['weight','height'], inplace=True)
        return self

    def process_bp(self):
        def classify_bp(sbp, dbp):
            if sbp < 120 and dbp < 80:
                return 'normal'
            elif 120 <= sbp <= 129 and dbp < 80:
                return 'above_normal'
            elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
                return 'first'
            elif sbp >= 140 or dbp >= 90:
                return 'second'

        self.df['bp_category'] = [classify_bp(s, b) for s, b in zip(self.df['ap_hi'], self.df['ap_lo'])]
        self.df = pd.concat([self.df, pd.get_dummies(self.df['bp_category'], prefix='bp', dtype=int)], axis=1)
        self.df.drop(columns=['ap_hi','ap_lo','bp_category'], inplace=True)
        return self

    def transform(self):
        (self
         .delete_strange()
         # .one_hot_from_map('cholesterol', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'chole')
         # .one_hot_from_map('gluc', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'gluc')
         # .process_age()
         # .process_gender()
         # .process_bmi()
         # .process_bp()
        )
        return self.df


# ✅ 提供一個函式，讓其他檔案可以直接 import 使用
def load_and_preprocess(path="data/Cardiovascular Disease Dataset.csv"):
    data = pd.read_csv(path, sep=';')
    features = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    X = data[features]
    y = data['cardio']

    preprocessor = CardioPreprocessor(X)
    X_processed = preprocessor.transform()
    XX_processed, y = X_processed.align(y, axis=0, join='inner')

    return X_processed, y


if __name__ == "__main__":
    # 測試用
    X_processed, y = load_and_preprocess()
    pd.set_option('display.max_columns', None)
    print(X_processed.head(20))
    print(X_processed.info())
    print(X_processed.shape, y.shape)
    import numpy as np

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    print(X_processed.index.equals(y.index))
    summary = X_processed.groupby(y).mean().T
    summary['diff'] = summary[1] - summary[0]
    print(summary.sort_values('diff', ascending=False))
