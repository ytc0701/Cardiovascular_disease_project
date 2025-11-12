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
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
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

    def find_best_threshold_by_youden_j(self, y_true, probs):
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
        return best_thresh

    def find_best_threshold_by_f1(self, y_true, probs):
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        f1_scores = np.nan_to_num(2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8))
        best_thresh = thresholds[np.argmax(f1_scores)]
    
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, label="F1-score")
        plt.axvline(best_thresh, color='r', linestyle='--', label=f"Best Threshold = {best_thresh:.4f}")
        plt.xlabel("Threshold")
        plt.ylabel("F1-score")
        plt.title("F1-score vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        return best_thresh


    # 3️⃣ 神經網路
    def train_nn(self):
        # 建構模型
        model = models.Sequential([
            layers.Input(shape=(self.X_train.shape[1],)),
            layers.Dense(12, activation='relu'),
            layers.Dense(9, activation='relu'),
            layers.Dense(6, activation='relu'),
            layers.Dense(3, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
        # EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
        # 模型訓練
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=64,
            callbacks=[es],
            verbose=1,
            validation_data=(self.X_test, self.y_test)
        )
    
        # 預測機率
        probs = model.predict(self.X_test).reshape(-1)
    
        # 找出最佳 threshold
        best_j_thresh = self.find_best_threshold_by_youden_j(self.y_test, probs)
        best_f1_thresh = self.find_best_threshold_by_f1(self.y_test, probs)
    
        print(f"\nBest threshold by Youden’s J index: {best_j_thresh:.4f}")
        print(f"Best threshold by F1-score: {best_f1_thresh:.4f}")
    
        # 使用最佳 F1 threshold 預測
        preds = (probs > best_j_thresh).astype(int)
        acc = accuracy_score(self.y_test, preds)
        print(f"\n[Threshold = {best_f1_thresh:.4f}] Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, preds))
        
        
        # 混淆矩陣
        cm = confusion_matrix(self.y_test, preds)
        cm_sum = np.sum(cm)
        cm_perc = cm / cm_sum * 100
        
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            cm,
            cmap='Reds',
            cbar=False,
            linewidths=1,
            linecolor='white',
            square=True
        )
        
        # 手動標註每個格子
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                c = cm[i, j]
                p = cm_perc[i, j]
                text = f"{c}\n({p:.1f}%)"
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color='black', fontsize=12, fontweight='bold')
        
        plt.title("Neural Network - Confusion Matrix", fontsize=16, weight='bold')
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks([0.5, 1.5], ['0', '1'])
        plt.yticks([0.5, 1.5], ['0', '1'])
        plt.tight_layout()
        plt.show()


    
        # ROC 曲線繪製
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        roc_auc = auc(fpr, tpr)
        print(f"Neural Network AUC: {roc_auc:.4f}")
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Neural Network - ROC Curve")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
        return model, self.X_test, self.y_test, best_j_thresh




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
            annot_kws={"size": 12, "weight": "bold"}
        )

        for text, color in zip(ax.texts, ax.collections[0].get_facecolors()):
            r, g, b, _ = color
            brightness = (r + g + b) / 3
            text.set_color('black' if brightness > 0.6 else 'white')

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
        else:
            probs = None

        if probs is not None:
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            roc_auc = auc(fpr, tpr)
            print(f"{model_name} AUC: {roc_auc:.4f}")
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{model_name} - ROC Curve")
            plt.legend(loc='lower right')
            plt.show()

    
def plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="blue"):
    """
    繪製所有特徵兩兩組合的錯誤預測散點圖矩陣（正方形比例、點小、字小、避免重疊）
    label_filter: "blue" → 只畫真實標籤為 1；"red" → 只畫真實標籤為 0；"all" → 同時畫
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    # 預測
    probs = model.predict(X_test).reshape(-1)
    preds = (probs > threshold).astype(int)

    # 錯誤樣本
    wrong_idx = X_test.index[preds != y_test]
    X_wrong = X_test.loc[wrong_idx]
    y_wrong = y_test.loc[wrong_idx]

    # 所有特徵組合（10個特徵 → 45組）
    selected_features = X_test.columns[:10]
    pairs = list(itertools.combinations(selected_features, 2))
    n = len(pairs)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten()

    for i, (f1, f2) in enumerate(pairs):
        ax = axes[i]
        # 藍=0 紅=1
        if label_filter == "red":
            mask = y_wrong == 1
            color = 'red'
        elif label_filter == "blue":
            mask = y_wrong == 0
            color = 'blue'
        else:  # all
            mask = np.ones(len(y_wrong), dtype=bool)
            color = ['red' if label == 1 else 'blue' for label in y_wrong]

        X_filtered = X_wrong.loc[mask]
        jitter_x = X_filtered[f1] + np.random.normal(0, 0.01, size=len(X_filtered))
        jitter_y = X_filtered[f2] + np.random.normal(0, 0.01, size=len(X_filtered))

        ax.scatter(jitter_x, jitter_y, alpha=0.3, c=color, s=3)

        ax.set_xlabel(f1, fontsize=5)
        ax.set_ylabel(f2, fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])

        # 強制正方形比例
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal', adjustable='box')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.0)
    plt.suptitle("Wrong point scatter matrix (colored by true label)", fontsize=5, y=1.02)
    plt.show()




def summarize_feature_value_by_true_label_in_errors(model, X_test, y_test, threshold, top_n=10, visualize=True):
    """
    分析錯誤與正確預測樣本中，各特徵在 y=0 和 y=1 下的平均值與差異
    - 根據差異絕對值排序
    - 顯示 y 分布
    - 可選擇是否進行 3D 視覺化
    """
    import numpy as np
    import pandas as pd
    from data_washing import plot_3d_scatter

    # 預測
    probs = model.predict(X_test).reshape(-1)
    preds = (probs > threshold).astype(int)

    # 錯誤與正確樣本
    wrong_idx = X_test.index[preds != y_test]
    correct_idx = X_test.index[preds == y_test]

    X_wrong = X_test.loc[wrong_idx]
    y_wrong = y_test.loc[wrong_idx]

    X_correct = X_test.loc[correct_idx]
    y_correct = y_test.loc[correct_idx]

    def analyze_subset(X_subset, y_subset, label="錯誤樣本"):
        print(f"\n📊 {label}中 y 分布：{dict(zip(*np.unique(y_subset, return_counts=True)))}")

        summary = X_subset.groupby(y_subset).mean().T
        summary['diff'] = summary[1] - summary[0]
        summary_sorted = summary.reindex(summary['diff'].abs().sort_values(ascending=False).index)

        print(f"\n{label}中各特徵的平均值（y=1 - y=0），依差異絕對值排序：")
        print(summary_sorted.head(top_n))

        if visualize:
            try:
                plot_3d_scatter(
                    X_subset.assign(
                        ap_hi=X_subset['ap_hi'],
                        ap_lo=X_subset['ap_lo'],
                        age_years=X_subset['age_years']
                    ),
                    y_subset
                )
            except Exception as e:
                print(f"⚠️ 3D 視覺化失敗：{e}")

    # 分析錯誤樣本
    analyze_subset(X_wrong, y_wrong, label="錯誤樣本")

    # 分析正確樣本
    analyze_subset(X_correct, y_correct, label="正確樣本")





    
 
    
if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(X.shape,y.shape)
    print(X.head(),y.head())

    trainer = ModelTrainer(X, y)

    # 調參 & 評估
    # trainer.tune_random_forest()
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
    model, X_test, y_test, threshold = trainer.train_nn()
    # trainer.train_knn(5)
    # trainer.train_naive_bayes()
    # plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="blue")  # 只畫藍點
    # plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="red")   # 只畫紅點
    plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="all")   # 同時畫紅藍點
    summarize_feature_value_by_true_label_in_errors(model, X_test, y_test, threshold, top_n=10)

