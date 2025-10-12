from data_washing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np

# XGBoost / LightGBM / CatBoost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Neural Network
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def evaluate_model(self, model, name):
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, preds))

        # 新增可視化
        self.visualize_model_performance(model, name)
        return model

    # 1️⃣ 線性模型
    def train_logistic(self):
        return self.evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

    def train_linear_svm(self):
        return self.evaluate_model(LinearSVC(max_iter=2000), "Linear SVM")

    # 2️⃣ 樹模型
    def train_decision_tree(self):
        return self.evaluate_model(DecisionTreeClassifier(random_state=42), "Decision Tree")

    def train_random_forest(self):
        return self.evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

    def train_xgboost(self):
        return self.evaluate_model(XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
                                                 colsample_bytree=0.8, random_state=42, eval_metric="logloss"),"XGBoost")

    def train_lightgbm(self):
        return self.evaluate_model(LGBMClassifier(random_state=42), "LightGBM")

    def train_catboost(self):
        return self.evaluate_model(CatBoostClassifier(verbose=0, random_state=42), "CatBoost")

    # 3️⃣ 神經網路
    def train_nn(self):
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              activation='relu',
                              solver='adam',
                              max_iter=500,
                              random_state=42)
        return self.evaluate_model(model, "Neural Network (MLP)")

    # 4️⃣ 其他模型
    def train_knn(self, k):
        return self.evaluate_model(KNeighborsClassifier(n_neighbors=k), f"kNN (k={k})")

    def train_naive_bayes(self):
        return self.evaluate_model(BernoulliNB(), "Naive Bayes")


    # ========================
    # 🧠 超參數搜尋模組
    # ========================
    def tune_model(self, model, param_grid, model_name, n_iter=20):
        """使用 RandomizedSearchCV 尋找最佳參數"""
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        search.fit(self.X_train, self.y_train)
        print(f"\n🔍 {model_name} Best Params: {search.best_params_}")
        print(f"Best CV Accuracy: {search.best_score_:.4f}")
        return search.best_estimator_

    # ========================
    # 各模型的超參數搜尋
    # ========================
    def tune_random_forest(self):
        param_grid = {
            'n_estimators': [500],
            'max_depth': [10],
            'min_samples_split': [10],
            'min_samples_leaf': [4],
            'max_features': ['sqrt']
        }
        #Random Forest Best Params: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 500}Best CV Accuracy: 0.7323

        model = RandomForestClassifier(random_state=42)
        best_model = self.tune_model(model, param_grid, "Random Forest")
        return self.evaluate_model(best_model, "Random Forest (Tuned)")

    def tune_xgboost(self):
        param_grid = {
            'n_estimators': [1200],
            'learning_rate': [0.005],
            'max_depth': [3],
            'subsample': [0.3],
            'colsample_bytree': [0.8],
        }
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        best_model = self.tune_model(model, param_grid, "XGBoost")
        return self.evaluate_model(best_model, "XGBoost (Tuned)")
        #🔍 XGBoost Best Params: {'colsample_bytree': 0.8, 'learning_rate': 0.005, 'max_depth': 3, 'n_estimators': 1200, 'subsample': 0.3} Best CV Accuracy: 0.6976

    def tune_lightgbm(self):
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05],
            'num_leaves': [15],
            'max_depth': [-1],
            'min_child_samples': [30],
        }
        model = LGBMClassifier(random_state=42)
        best_model = self.tune_model(model, param_grid, "LightGBM")
        return self.evaluate_model(best_model, "LightGBM (Tuned)")

    def tune_catboost(self):
        param_grid = {
            'iterations': [600],
            'depth': [4],
            'learning_rate': [0.01],
            'l2_leaf_reg': [ 3,5],
        }
        model = CatBoostClassifier(verbose=0, random_state=42)
        best_model = self.tune_model(model, param_grid, "CatBoost")
        return self.evaluate_model(best_model, "CatBoost (Tuned)")

    def tune_logistic(self):
        param_grid = {
            'C': np.logspace(1, 1, ),  # 0.001 → 1000
            'penalty': ['l1'],  # L1 for feature selection, L2 for stability
            'solver': ['saga'],  # 支援 L1 與 L2，適用大資料
            'max_iter': [3000]
        }
        model = LogisticRegression(random_state=42)
        best_model = self.tune_model(model, param_grid, "Logistic Regression")
        return self.evaluate_model(best_model, "Logistic Regression (Tuned)")



    def visualize_model_performance(self, model, model_name):
        """顯示混淆矩陣與ROC曲線"""
        preds = model.predict(self.X_test)
        probs = None

        # 混淆矩陣
        cm = confusion_matrix(self.y_test, preds)
        cm_sum = np.sum(cm)
        cm_perc = cm / cm_sum * 100

        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape

        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                annot[i, j] = f"{c}\n({p:.1f}%)"

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=False,
            linewidths=1,
            linecolor='white',
            square=True,
            annot_kws={"size": 13, "weight": "bold"}
        )

        # ✅ 修正版 - 強制顯示所有文字並確保對比明顯
        for text in ax.texts:
            # 取得顏色對比
            r, g, b, _ = ax.collections[0].get_facecolor()[0]
            brightness = (r + g + b) / 3
            text.set_color('black' if brightness > 0.6 else 'white')

        # 再覆蓋一次標籤（防止被遮住）
        for text in ax.texts:
            text.set_visible(True)

        plt.title(f"{model_name} - Confusion Matrix", fontsize=16, weight='bold')
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.show()

        # 如果模型有 predict_proba 或 decision_function，可畫 ROC 曲線
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(self.X_test)

        if probs is not None:
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{model_name} - ROC Curve")
            plt.legend(loc='lower right')
            plt.show()

if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(X.shape,y.shape)

    trainer = ModelTrainer(X, y)

    # 調參 & 評估
    trainer.tune_random_forest()
    # trainer.tune_xgboost()
    # trainer.tune_lightgbm()
    # trainer.tune_catboost()
    # trainer.tune_logistic()

    # # 逐一訓練模型
    # trainer.train_logistic()
    # trainer.train_linear_svm()
    # trainer.train_decision_tree()
    # trainer.train_random_forest()
    # trainer.train_xgboost()
    # trainer.train_lightgbm()
    # trainer.train_catboost()
    # trainer.train_nn()
    # trainer.train_knn(5)
    # trainer.train_naive_bayes()
