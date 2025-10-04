import pandas as pd
#
# # 讀取資料,使用 sep=';' 告訴 pandas 每個欄位是用分號分隔
# data = pd.read_csv('Cardiovascular Disease Dataset.csv', sep=';')
# # print(data.head())
# # print(data.tail())
# # print(data.shape)
# print(data.info())
# print(data.describe())
# # 查看每個欄位有多少種獨特值
# unique_counts = data.nunique()
# print(unique_counts)
#
# features = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
# X = data[features]
#
# y = data['cardio']
#
#
# def one_hot_from_map(df, col, mapping, prefix):
#     """
#     將數值欄位依 mapping 轉換成多個 0/1 欄位
#     mapping: {原值: 新欄位名稱}
#     """
#     for val, name in mapping.items():
#         df[f"{prefix}_{name}"] = (df[col] == val).astype(int)
#     df.drop(columns=[col], inplace=True)
#     return df
#
# X = one_hot_from_map(X, 'cholesterol', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'chole')
# X = one_hot_from_map(X, 'gluc', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'gluc')
#
# def categorize_and_onehot(df, col, bins, labels, prefix):
#     df[f"{col}_group"] = pd.cut(df[col], bins=bins, labels=labels, right=True)
#     dummies = pd.get_dummies(df[f"{col}_group"], prefix=prefix, dtype=int)
#     df = pd.concat([df, dummies], axis=1)
#     df.drop(columns=[col, f"{col}_group"], inplace=True)
#     return df
#
# age_bins = [28, 34, 44, 54, 64]  # 左開右閉，根據美國心臟協會 (AHA) 與臨床研究常用分層
# age_labels = ['young_adult','early_middle','middle','late_middle']
# X['age_years'] = X['age'] // 365
# X = categorize_and_onehot(X, 'age_years', age_bins, age_labels, 'age')
# X.drop(columns=['age'], inplace=True)
#
# bmi_bins = [0, 18.5, 25, 30, 100]
# bmi_labels = ['underweight','normal','overweight','obese']
# X['BMI'] = X['weight'] / ((X['height']/100)**2)
# X = categorize_and_onehot(X, 'BMI', bmi_bins, bmi_labels, 'BMI')
# X.drop(columns=['weight','height'], inplace=True)
#
# #
# # def categorize_age(age):
# #     if 29 <= age <= 34:
# #         return "young_adult"
# #     elif 35 <= age <= 44:
# #         return "early_middle"
# #     elif 45 <= age <= 54:
# #         return "middle"
# #     elif 55 <= age <= 64:
# #         return "late_middle"
# #
# # # 建立臨時欄位用於分類
# # X['age_group'] = X['age_years'].apply(categorize_age)
# #
# # # One-hot encoding 直接產生 0/1 欄位
# # age_dummies = pd.get_dummies(X['age_group'], prefix='age', dtype=int)
# #
# # # 合併到原本的 dataframe
# # X = pd.concat([X, age_dummies], axis=1)
# #
# # # 刪除原本的文字標籤欄位（可選）
# # X.drop(columns=['age_group','age_years'], inplace=True)
#
#
# X['male'] = (X['gender'] == 1).astype(int)
# X['female'] = 1 - X['male']
# X = X.drop(columns=['gender'])
#
#
# # def categorize_bmi(bmi):
# #     if bmi < 18.5:
# #         return "bmi_underweight"
# #     elif 18.5 <= bmi < 25:
# #         return "bmi_normal"
# #     elif 25 <= bmi < 30:
# #         return "bmi_overweight"
# #     else:  # bmi >= 30
# #         return "bmi_obese"
# #
# # # 建立臨時欄位
# # X['bmi_group'] = X['BMI'].apply(categorize_bmi)
# #
# # # One-hot encoding 產生 0/1 欄位
# # bmi_dummies = pd.get_dummies(X['bmi_group'], prefix='BMI', dtype=int)
# #
# # # 合併到原本 dataframe
# # X = pd.concat([X, bmi_dummies], axis=1)
# #
# # # 刪除文字欄位（可選）
# # X.drop(columns=['bmi_group','BMI'], inplace=True)
#
# # X['chole_nor'] = (X['cholesterol'] == 1).astype(int)
# # X['chole_abnor'] = (X['cholesterol'] == 2).astype(int)
# # X['chole_wellabnor'] = (X['cholesterol'] == 3).astype(int)
# # X = X.drop(columns=['cholesterol'])
# #
# # X['gluc_nor'] = (X['gluc'] == 1).astype(int)
# # X['gluc_abnor'] = (X['gluc'] == 2).astype(int)
# # X['gluc_wellabnor'] = (X['gluc'] == 3).astype(int)
# # X = X.drop(columns=['gluc'])
#
#
# # 假設你的 DataFrame 叫做 df，且有 ap_hi (收縮壓) 和 ap_lo (舒張壓)
#
# def classify_bp_series(sbp, dbp):
#     if sbp < 120 and dbp < 80:
#         return 'normal'
#     elif 120 <= sbp <= 129 and dbp < 80:
#         return 'above_normal'
#     elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
#         return 'first'
#     elif sbp >= 140 or dbp >= 90:
#         return 'second'
#
#
# X['bp_category'] = [classify_bp_series(s,b) for s,b in zip(X['ap_hi'], X['ap_lo'])]
# X = pd.concat([X, pd.get_dummies(X['bp_category'], prefix='bp', dtype=int)], axis=1)
# X.drop(columns=['ap_hi','ap_lo','bp_category'], inplace=True)
#
# pd.set_option('display.max_columns', None)
#
# print(X.head(20))

import pandas as pd

class CardioPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

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
         .one_hot_from_map('cholesterol', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'chole')
         .one_hot_from_map('gluc', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'gluc')
         .process_age()
         .process_gender()
         .process_bmi()
         .process_bp()
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

    return X_processed, y


if __name__ == "__main__":
    # 測試用
    X_processed, y = load_and_preprocess()
    pd.set_option('display.max_columns', None)
    print(X_processed.head(20))