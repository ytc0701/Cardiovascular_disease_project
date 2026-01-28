import pandas as pd

df = pd.read_csv("data/Cardiovascular Disease Dataset.csv", sep=";")
from data_washing import CardioPreprocessor
df = CardioPreprocessor(df).delete_strange().df


pd.set_option('display.max_columns', None)
print(pd.concat([df.head(5), df.sample(5), df.tail(5)]))
print(df.describe(include='all').T)
df.info()
pd.DataFrame([ df.nunique(), df.dtypes ], index=['Unique Values', 'Data Types'])
list(df.columns)
TARGET = 'cardio'
INPUT_FEATURES = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
df.duplicated().sum() #重複序列總數===>0

#查看有無na值及na值比例==>無 0%
pd.DataFrame(
    {
    "Count" : df.isna().sum().sort_values(ascending=False),
    "percent %" : 100.0 * df.isna().sum().sort_values(ascending=False) / len(df),
    }
).T

def drop_unnwanted_features(df, features_to_drop):
    print(f"Dropping features: {features_to_drop}")

    if features_to_drop in list(df.columns):
        df = df.drop(columns=features_to_drop)

    if features_to_drop in INPUT_FEATURES:
        INPUT_FEATURES.remove(features_to_drop)
    return df

df = drop_unnwanted_features(df, 'id')

list(df.columns)

#EDA & Dataset Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("Using style:", plt.style.available[12])
plt.style.use(plt.style.available[12])

plt.figure(figsize=(3,3))
plt.title(f"Count plot: {TARGET}")
sns.countplot(df, x=TARGET)
plt.show()#There is no bias in target column
#%%
#畫散佈圖

ROW_WIDTH = 6
temp = 0

for num_feature in INPUT_FEATURES:
    if temp == 0:
        plt.figure(figsize=(27,10))

    plt.subplot(2,ROW_WIDTH, temp + 1)
    plt.title(f"Hist plot: {num_feature} (all categories)")
    sns.kdeplot(df, x=num_feature, )

    plt.axvline(x= df[num_feature].min() , label="min", color='r', linestyle='--', alpha = 0.3)
    plt.axvline(x= df[num_feature].mean() , label="mean", color='g', linestyle='--', alpha = 0.3)
    plt.axvline(x= df[num_feature].max() , label="max", color='b', linestyle='--', alpha = 0.3)

    plt.legend()


    plt.subplot(2,ROW_WIDTH, (temp + 1) + ROW_WIDTH)
    plt.title(f"Hist plot: {num_feature} (for each category)")
    sns.kdeplot(df, x=num_feature, hue=TARGET, palette='tab10')
    if temp == ROW_WIDTH - 1:
        plt.show()

    temp = (temp + 1) % ROW_WIDTH
    
if temp != ROW_WIDTH - 1:
    plt.show()
#%%
#畫個特徵對有無疾病之Box_Polt 盒鬚圖（又稱箱型圖）

import math

N_COLS = 3
N_ROWS = math.ceil(len(INPUT_FEATURES) / N_COLS)

plt.figure(figsize=(10,8))

for i, feature in enumerate(INPUT_FEATURES):
    plt.subplot(N_ROWS, N_COLS, i+1)
    plt.title(f"Box plot: {feature}")
    sns.boxplot(df, x=TARGET, y=feature)
    plt.ylabel("")
    plt.xlabel('')

plt.show()
#%%
#畫個特徵的盒鬚圖（Box Plot）

def plot_box(features, hue):
    ROW_WIDTH = 6
    temp = 0

    for num_feature in features:
        if temp == 0:
            plt.figure(figsize=(25,5))

        plt.subplot(1,ROW_WIDTH, temp + 1)
        plt.title(f"Box plot: {num_feature}")
        sns.boxplot(df, x=num_feature, hue=hue, palette='rocket')

        if temp == ROW_WIDTH - 1:
            plt.show()

        temp = (temp + 1) % ROW_WIDTH

    if temp != ROW_WIDTH - 1:
        plt.show()
        
plot_box(INPUT_FEATURES, TARGET)
#%%

# def IQR(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)

#     IQR = Q3 - Q1

#     min_v = Q1 - 1.5 * IQR
#     max_v = Q3 + 1.5 * IQR

#     return series.clip(lower=min_v, upper=max_v)#將超出上下限的值裁切為邊界值，避免極端值影響分析。

# exclude = ['gluc','alco','smoke','cholesterol','active']
# print(df[TARGET].unique())

# for num_feature in [f for f in INPUT_FEATURES if f not in exclude]:
#     for target_category in df[TARGET].unique():
#         mask = df[TARGET] == target_category
#         df.loc[mask, num_feature] = IQR(df.loc[mask, num_feature])#分有疾病跟沒疾病的族群來去除離群值
        
# plot_box(INPUT_FEATURES, TARGET)
#%%
INPUT_FEATURES
df.head(5)
df['age_years'] = (df['age'] / 365).round().astype(int)
# df['height'] = df['height'] / 100
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['health_index'] = (df['active'] * 1) - (df['smoke'] * 0.5) - (df['alco'] * 0.5)
df['cholesterol_gluc_interaction'] = df['cholesterol'] * df['gluc']

# 平均動脈壓 (Mean Arterial Pressure)
df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

# 高血壓分級 (Hypertension Stage)
def hypertension_stage(row):
    if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
        return 2  # 高血壓
    elif row['ap_hi'] >= 130 or row['ap_lo'] >= 85:
        return 1  # 前期高血壓
    else:
        return 0  # 正常
df['hypertension_stage'] = df.apply(hypertension_stage, axis=1)

# 代謝症候群指標 (Metabolic Syndrome Indicator)
def metabolic_syndrome(row):
    count = 0
    if row['bmi'] >= 30: count += 1
    if row['ap_hi'] >= 130 or row['ap_lo'] >= 85: count += 1
    if row['cholesterol'] >= 2: count += 1
    if row['gluc'] >= 2: count += 1
    return count
df['metabolic_syndrome'] = df.apply(metabolic_syndrome, axis=1)

# 綜合風險分數 (Composite Cardiovascular Risk Index)
df['risk_index'] = (
    (df['age_years'] > 50).astype(int) +
    (df['bmi'] > 30).astype(int) +
    (df['pulse_pressure'] > 60).astype(int) +
    (df['cholesterol'] >= 2).astype(int) +
    (df['gluc'] >= 2).astype(int) +
    df['smoke'] +
    df['alco']
)

# 性別與生活習慣交互項 (Gender × Lifestyle)
df['gender_smoke_interaction'] = df['gender'] * df['smoke']
df['gender_alco_interaction'] = df['gender'] * df['alco']




new_features = [
    'age_years',
    'bmi',
    'pulse_pressure',
    'health_index',
    'cholesterol_gluc_interaction',
    'map',
    'hypertension_stage',
    'metabolic_syndrome',
    'risk_index',
    'gender_smoke_interaction',
    'gender_alco_interaction'
]

INPUT_FEATURES = INPUT_FEATURES + new_features

df = drop_unnwanted_features(df, 'age')
# df = drop_unnwanted_features(df, 'height')
df.columns

#%%
#Correlation Matrix
plt.figure(figsize=(30, 30))

plt.title("Input Features Correlation")

sns.heatmap(
    df[INPUT_FEATURES].corr(),
    annot=True,
    cmap='coolwarm',
    )

plt.show()
#%%
X = df.drop(TARGET, axis=1)
y = df.loc[:, TARGET]
print(X)
print(y)
from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()

X_stan = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(X_stan.shape)
print(X_stan)
# 存起來
joblib.dump(scaler, "scaler.pkl")

#%%
'''
F1 test (ANOVA)

F-value: Measures how much the means between groups differ relative to the variance within the groups. Higher = more likely the feature is important.

p-value: Probability that the observed difference is due to chance.

p < 0.05 → Statistically significant

p ≥ 0.05 → Not significant
'''

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(X_stan, y)

for i in range(len(INPUT_FEATURES)):
    print(f"{INPUT_FEATURES[i]:15s}: F-value = {f_values[i]:6.3f}, p-value = {p_values[i]:3.3f}")
X_stan = drop_unnwanted_features(X_stan, 'gender')
X_stan = drop_unnwanted_features(X_stan, 'alco')
X_stan = drop_unnwanted_features(X_stan, 'gender_alco_interaction')
X_stan = drop_unnwanted_features(X_stan, 'risk_index')
X_stan = drop_unnwanted_features(X_stan, 'pulse_pressure')
X_stan = drop_unnwanted_features(X_stan, 'height')
X_stan = drop_unnwanted_features(X_stan, 'smoke')
X_stan = drop_unnwanted_features(X_stan, 'active')

print(X_stan.shape)
#資料預處理結束
#%%
from training import ModelTrainer
trainer = ModelTrainer(X_stan,y)
# model, X_test, y_test, raw_probs = trainer.train_nn_with_optuna()
model, X_test, y_test, threshold, preds = trainer.train_nn()

# 儲存整個模型
model.save("cardio_model.h5")

#%%
#Dataset Splitting
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(
    X_stan,
    y,
    test_size=0.25,
    random_state= 42,
    stratify=df.loc[:, TARGET],
  )
print(f"{X_train.shape= }")
print(f"{y_train.shape= }")
print(f"{X_test.shape= }")
print(f"{y_test.shape= }")
y_train.value_counts()
y_test.value_counts()
list(X_train.columns)


#%%
#Model Definition & Training
import os
CACHE_MODELS_DIR_NAME = 'models_cache'
os.makedirs(CACHE_MODELS_DIR_NAME, exist_ok=True)
#%%
import os
import pickle
import re

class CustomModel:
    def __init__(self, name, model, extra_train_param=None):
        self.name = str(name)
        self.model = model
        self.extra_train_param = extra_train_param

        self.y_train_hat = None
        self.y_test_hat = None
        self.feature_names = None  # هنخزن الأعمدة هنا

        self.load()

    def fit(self, x_train, y_train):
        if not getattr(self, "trained", False):
            if self.extra_train_param is None:
                self.model.fit(x_train, y_train)
            else:
                self.model.fit(x_train, y_train, **self.extra_train_param)

            # نخزن الأعمدة
            self.feature_names = list(x_train.columns)

            self.trained = True
            self.save()

    def _align_features(self, X):
        """يتأكد إن X نفس الأعمدة اللي اتدرب عليها"""
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_names)

            if missing_cols:
                raise ValueError(f"Missing columns in input: {missing_cols}")
            if extra_cols:
                # ممكن بس نتجاهل الزيادة ونرتب الصح
                X = X[self.feature_names]
            else:
                X = X[self.feature_names]

        return X

    def prdict_on_train(self, x_train):
        if self.y_train_hat is None:
            x_train = self._align_features(x_train)
            self.y_train_hat = self.model.predict(x_train)

    def prdict_on_test(self, x_test):
        if self.y_test_hat is None:
            x_test = self._align_features(x_test)
            self.y_test_hat = self.model.predict(x_test)

    def save(self):
        file_name = re.sub(r'\W+', '_', str(self.name).lower())
        file_path = CACHE_MODELS_DIR_NAME + '/' + file_name + '.pickle'
        with open(file_path, 'wb') as f:
            pickle.dump(
                {
                    "model": self.model,
                    "trained": self.trained,
                    "feature_names": self.feature_names
                },
                f
            )

    def load(self):
        file_name = re.sub(r'\W+', '_', str(self.name).lower())
        file_path = CACHE_MODELS_DIR_NAME + '/' + file_name + '.pickle'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.trained = data["trained"]
                self.feature_names = data.get("feature_names", None)
        else:
            self.trained = False 

models_list = []

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


models_list.append(CustomModel("Logistic Regression", LogisticRegression()))

models_list.append(CustomModel("SVC rbf sigmoid", SVC( ))) #بيطول بغباء

models_list.append(CustomModel("Decision Tree max-depth=7", DecisionTreeClassifier(max_depth=7 )))
models_list.append(CustomModel("Decision Tree max-depth=10", DecisionTreeClassifier(max_depth=10,min_samples_split=7,min_samples_leaf=3,criterion='gini' )))

models_list.append(CustomModel("Random Forest, trees=3", RandomForestClassifier(n_estimators=3 )))
models_list.append(CustomModel("Random Forest, trees=5", RandomForestClassifier(n_estimators=5 )))


models_list.append(CustomModel("Ada Boost estimator=5 max_depth=2", AdaBoostClassifier(n_estimators=5, estimator=RandomForestClassifier(max_depth=2))))

models_list.append(CustomModel("Gradient Boosting estimator=3", GradientBoostingClassifier(n_estimators=3)))


for i, model in enumerate(models_list):
    print(f'{i+1:3d}/{len(models_list)}. Train {model.name}')
    model.fit(X_train,y_train)


#%%
#Model Prediction & Evaluation

for i, model in enumerate(models_list):
    print(f'{i+1:3d}/{len(models_list)}. Predict {model.name} on train data')
    model.prdict_on_train(X_train)
for i, model in enumerate(models_list):
    print(f'{i+1:3d}/{len(models_list)}. Predict {model.name} on test data')
    model.prdict_on_test(X_test)
#%%
#Evaluation
evaluation_dataset = []
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

for i, model in enumerate(models_list):
    print(f'{i+1:02d}/{len(models_list)}. Evaluate {model.name}')

    accuracy_score_train = accuracy_score( y_train, model.y_train_hat )
    accuracy_score_test = accuracy_score( y_test, model.y_test_hat )

    recall_score_train = recall_score( y_train, model.y_train_hat )
    recall_score_test = recall_score( y_test, model.y_test_hat )

    precision_score_train = precision_score( y_train, model.y_train_hat )
    precision_score_test = precision_score( y_test, model.y_test_hat )

    f1_score_train = f1_score( y_train, model.y_train_hat )
    f1_score_test = f1_score( y_test, model.y_test_hat)

    roc_auc_score_train = roc_auc_score( y_train, model.y_train_hat )
    roc_auc_score_test = roc_auc_score( y_test, model.y_test_hat)


    evaluation_dataset.append({
        "model": model.name, 'accuracy_score': accuracy_score_train,
        'recall_score': recall_score_train, 'f1_score':f1_score_train,
        'roc_auc_score':roc_auc_score_train, 'precision_score':precision_score_train,
        'data':'train'}
    )

    evaluation_dataset.append({
        "model": model.name, 'accuracy_score': accuracy_score_test,
        'recall_score': recall_score_test, 'f1_score':f1_score_test,
        'roc_auc_score':roc_auc_score_test, 'precision_score':precision_score_test,
        'data':'test'}
    )


    # print(evaluation_dataset[-2])
    # print(evaluation_dataset[-1])
    print('\n')


evaluation_dataset = pd.DataFrame(evaluation_dataset)

evaluation_dataset.sort_values('f1_score', ascending=False)

print("Best Model in F1 Score test")
evaluation_dataset[evaluation_dataset['data']=='test'].sort_values('f1_score',ascending=False).iloc[0,:]

print("Best Model in ROC-AUC test")
evaluation_dataset[evaluation_dataset['data']=='test'].sort_values('roc_auc_score',ascending=False).iloc[0,:]

import numpy as np

for metric in evaluation_dataset.columns:
    if metric in ['model', 'data']:
        continue

    plt.figure(figsize=(15,4))
    plt.title(f"{metric}")
    sns.barplot(evaluation_dataset, x='model',y=metric, hue='data')
    plt.xticks(rotation = 90)
    plt.yticks(np.linspace(0,1,11))
    plt.show()

#%%
#Caching快取
evaluation_dataset.to_csv("eval_dataset.csv")

CAHCE_Y_VS_Y_HAT_DIR_NAME = 'models_predictions'
os.makedirs(CAHCE_Y_VS_Y_HAT_DIR_NAME, exist_ok=True)

for model in models_list:
    pad_width = len(y_train) - len(y_test)

    pd.DataFrame(
        {
            'actual Y train' : y_train,
            'predict Y train' : model.y_train_hat,
            'Train Diff':  y_train - model.y_train_hat,
            'actual Y test' : np.pad(y_test, (0, pad_width), mode='constant', constant_values=-1),
            'predict Y test' : np.pad(model.y_test_hat, (0, pad_width), mode='constant', constant_values=-1),
            'Test Diff':  np.pad(y_test - model.y_test_hat, (0, pad_width), mode='constant'),
        }
    ).to_csv(f"{CAHCE_Y_VS_Y_HAT_DIR_NAME}/{model.name}.csv")























