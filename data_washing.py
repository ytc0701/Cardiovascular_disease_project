import pandas as pd
import matplotlib.pyplot as plt

class CardioPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def delete_strange(self):
        df = self.df
        # 年齡限制
        df = df[(df['age'] / 365 >= 0) & (df['age'] / 365 <= 150)]
        # 身高體重
        df = df[(df['height'] >= 120) & (df['height'] <= 300)]
        df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        # 血壓範圍
        df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 200)]
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
        # age_bins = [28, 34, 44, 54, 64]
        # age_labels = ['young_adult','early_middle','middle','late_middle']
        self.df['age_years'] = self.df['age'] // 365
        # self.categorize_and_onehot('age_years', age_bins, age_labels, 'age')
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
        # self.categorize_and_onehot('BMI', bmi_bins, bmi_labels, 'BMI')
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

    def standardize(self):
        from sklearn.preprocessing import StandardScaler
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self

    def transform(self):
        (self
         .delete_strange()
         # .one_hot_from_map('cholesterol', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'chole')
         # .one_hot_from_map('gluc', {1:'nor', 2:'abnor', 3:'wellabnor'}, 'gluc')
         .process_age()
         # .process_gender()
         .process_bmi()
         # .process_bp()
         .standardize()
        )
        # self.df = self.df.drop(columns=['gender', 'alco', 'smoke','active'])
        return self.df
    
    


# ✅ 提供一個函式，讓其他檔案可以直接 import 使用
def load_and_preprocess(path="data/Cardiovascular Disease Dataset.csv"):
    data = pd.read_csv(path, sep=';')
    features = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    X = data[features]
    y = data['cardio']

    preprocessor = CardioPreprocessor(X)
    X_processed = preprocessor.transform()
    X_processed, y = X_processed.align(y, axis=0, join='inner')  # ✅ 確保 index 對齊

    return X_processed, y

def plot_3d_scatter(X, y, x_col='ap_hi', y_col='ap_lo', z_col='age_years'):
    """
    在三維空間中繪製散點圖，三個軸分別為 x_col, y_col, z_col。
    點的顏色根據 y（cardio）分類。
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 根據 y 分類顏色 藍=0 紅=1
    colors = ['blue' if label == 0 else 'red' for label in y]

    ax.scatter(X[x_col], X[y_col], X[z_col], c=colors, alpha=0.6, s=20)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title("3D Scatter Plot: ap_hi vs ap_lo vs age_years")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    X_processed, y = load_and_preprocess()
    pd.set_option('display.max_columns', None)
    print("X_processed\n",X_processed.head(20))
    print("X_processed.info()\n",X_processed.info())
    print("shape\n",X_processed.shape, y.shape)
    import numpy as np

    #確認y中0跟1的數量
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    # 確認X跟y的index是否相符
    print(X_processed.index.equals(y.index))

    #計算每個類別（0 和 1）在各個特徵上的平均值。
    summary = X_processed.groupby(y).mean().T
    summary['diff'] = summary[1] - summary[0]
    print(summary.sort_values('diff', ascending=False))
    plot_3d_scatter(
        X_processed.assign(
            ap_hi= X_processed['ap_hi'],
            ap_lo= X_processed['ap_lo'],
            age_years= X_processed['age_years'] 
        ),
        y
    )
