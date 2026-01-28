from data_washing import load_and_preprocess,plot_3d_scatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import  GridSearchCV

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# XGBoost / LightGBM / CatBoost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall


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

    def tune_linear_svm(self):
        param_grid = {
            'C': [1],
            'max_iter': [2000],
            'loss': ['hinge', 'squared_hinge']
        }
        model = LinearSVC(random_state=42)
        best_model = self.tune_model(model, param_grid, "Linear SVM")
        return self.evaluate_model(best_model, "Linear SVM (Tuned)")

    # 2️⃣ 樹模型
    def tune_decision_tree(self):
        param_grid = {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        model = DecisionTreeClassifier(random_state=42)
        best_model = self.tune_model(model, param_grid, "Decision Tree")
        return self.evaluate_model(best_model, "Decision Tree (Tuned)")

    # 4️⃣ 其他模型
    def tune_knn(self):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        model = KNeighborsClassifier()
        best_model = self.tune_model(model, param_grid, "kNN")
        return self.evaluate_model(best_model, "kNN (Tuned)")

    def tune_naive_bayes(self):
        param_grid = {
            'alpha': [0.01, 0.05 ,0.1],
            'binarize': [ 1.0, 2.0]
        }
        model = BernoulliNB()
        best_model = self.tune_model(model, param_grid, "Naive Bayes")
        return self.evaluate_model(best_model, "Naive Bayes (Tuned)")


    def tune_random_forest(self):
        param_grid = {
            'n_estimators': [500],
            'max_depth': [15],
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
    # ========================
    # 🧠 超參數搜尋模組
    # ========================
    def tune_model(self, model, param_grid, model_name):
        """使用 GridSearchCV 尋找最佳參數"""
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

    def visualize_model_performance(self, model, model_name):
        """顯示混淆矩陣與 ROC 曲線（含清晰標註）"""
        preds = model.predict(self.X_test)
        probs = None
    
        # 混淆矩陣
        cm = confusion_matrix(self.y_test, preds)
        cm_sum = np.sum(cm)
        cm_perc = cm / cm_sum * 100
    
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            cm,
            cmap='Blues',
            cbar=False,
            linewidths=1,
            linecolor='white',
            square=True
        )
    
        # 手動標註每個格子（數值 + 百分比）
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                c = cm[i, j]
                p = cm_perc[i, j]
                text = f"{c}\n({p:.1f}%)"
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color='black', fontsize=12, fontweight='bold')
    
        plt.title(f"{model_name} - Confusion Matrix", fontsize=16, weight='bold')
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks([0.5, 1.5], ['0', '1'])
        plt.yticks([0.5, 1.5], ['0', '1'])
        plt.tight_layout()
        plt.show()
    
        # ROC 曲線
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(self.X_test)
    
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
            plt.grid(True)
            plt.show()
    

    # 3️⃣ 神經網路
    def train_nn(self):
        import matplotlib.pyplot as plt
        # 建構模型
        model = models.Sequential([
            layers.Input(shape=(self.X_train.shape[1],)),
            layers.Dense(9, activation='sigmoid'),
            layers.Dense(5, activation='tanh'),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Recall()])
    
        # EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
        # 模型訓練
        history = model.fit(
            self.X_train, self.y_train,
            epochs=1000,
            batch_size=64,
            callbacks=[es],
            verbose=1,
            validation_data=(self.X_test, self.y_test)
        )
    
        # 預測機率
        probs = model.predict(self.X_test).reshape(-1)
        print(probs[:100])

    
        # 找出最佳 threshold
        best_j_thresh = self.find_best_threshold_by_youden_j(self.y_test, probs)
        best_f1_thresh = self.find_best_threshold_by_f1(self.y_test, probs)
        
        print(f"\nBest threshold by Youden’s J index: {best_j_thresh:.4f}")
        print(f"Best threshold by F1-score: {best_f1_thresh:.4f}")
        
        


    
        # 混淆矩陣
        def plot_confusion_matrix(y_true, y_pred):
            import numpy as np
            import matplotlib.pyplot as plt
            cm = confusion_matrix(y_true, y_pred)
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
        
        
        # 原預測以0.5為Threshold
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(self.y_test, preds)
        print(f"\n[Origin probs] Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, preds))
        plot_confusion_matrix(self.y_test, preds)
        
        # 使用best_j_thresh為Threshold的預測
        best_j_preds = (probs > best_j_thresh).astype(int)
        best_j_acc = accuracy_score(self.y_test, best_j_preds)
        print(f"\n[Threshold = {best_j_thresh:.4f}] Accuracy: {best_j_acc:.4f}")
        print(classification_report(self.y_test, best_j_preds))
        plot_confusion_matrix(self.y_test, best_j_preds)
    
        # 使用最佳 F1 threshold為Threshold的預測
        best_f1_preds = (probs > best_f1_thresh).astype(int)
        best_f1_acc = accuracy_score(self.y_test, best_f1_preds)
        print(f"\n[Threshold = {best_f1_thresh:.4f}] Accuracy: {best_f1_acc:.4f}")
        print(classification_report(self.y_test, best_f1_preds))
        plot_confusion_matrix(self.y_test, best_f1_preds)
        
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
        
        
        


        import numpy as np
        import matplotlib.pyplot as plt
        
        # 建立 logit 範圍
        logits = np.linspace(-10, 10, 1000)
        
        # 套用 sigmoid
        sigmoid = 1 / (1 + np.exp(-logits))
        
        plt.figure(figsize=(8,6))
        
        # 畫 logit 曲線（直接顯示原始值）
        plt.plot(logits, logits, label="Logit (linear output)", color="orange")
        
        # 畫 sigmoid 曲線（壓縮到 0~1）
        plt.plot(logits, sigmoid, label="Sigmoid (probability)", color="blue")
        
        # 標記 logit=0 對應 sigmoid=0.5
        plt.axvline(0, color="red", linestyle="--", label="Logit=0 → Sigmoid=0.5")
        
        plt.title("Logit vs Sigmoid Curve")
        plt.xlabel("Logit value")
        plt.ylabel("Output")
        plt.legend()
        plt.grid(True)
        plt.show()

    
        return model, self.X_test, self.y_test, best_j_thresh, probs
    
        
        
    def train_nn_with_optuna(self, n_trials=200):
        import optuna
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
    
        def build_model(trial):
            model = models.Sequential()
            model.add(layers.Input(shape=(self.X_train.shape[1],)))
            # n_layers = trial.suggest_int("n_layers", 1, 4)
    
            # for i in range(2):
            num_hidden = trial.suggest_int("n_units_0", 2, 11)
            # activation = trial.suggest_categorical(f"activation_{i}", ["sigmoid","relu","elu","tanh"])
            model.add(layers.Dense(num_hidden, activation="sigmoid"))
            num_hidden = trial.suggest_int("n_units_1", 2, 11)
            model.add(layers.Dense(num_hidden, activation="tanh"))
            
            model.add(layers.Dense(1, activation="sigmoid"))
            return model
    
        def objective(trial):
            model = build_model(trial)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
            model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=100,
                batch_size=trial.suggest_categorical("batch_size", [32, 64]),
                callbacks=[es],
                verbose=0
            )
    
            probs = model.predict(self.X_test).reshape(-1)
            preds = (probs > 0.4).astype(int)
    
            # 計算 Recall 與 Accuracy
            recall = recall_score(self.y_test, preds)
            acc = accuracy_score(self.y_test, preds)
            print(f"Recall: {recall:.4f}")
            print(f"Accuracy: {acc:.4f}")
            # 複合指標：同時考慮 Recall 與 Accuracy
            score = 0.6 * recall + 0.4 * acc
            return score
    
        # Optuna 搜尋
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        print("\nBest trial:", best_params)
    
        # 使用最佳參數重建模型
        model = models.Sequential()
        model.add(layers.Input(shape=(self.X_train.shape[1],)))
        for i in range(1):
            units = best_params.get(f"n_units_{i}")
            act = best_params.get(f"activation_{i}")
            model.add(layers.Dense(units, activation=act))
        model.add(layers.Dense(1, activation="sigmoid"))
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
        model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=100,
            batch_size=best_params["batch_size"],
            callbacks=[es],
            verbose=1
        )
    
        raw_probs = model.predict(self.X_test).reshape(-1)
        preds = (raw_probs > 0.4).astype(int)
    
        # 最終輸出 Recall 與 Accuracy
        recall = recall_score(self.y_test, preds)
        acc = accuracy_score(self.y_test, preds)
    
        print(f"\nFinal Recall: {recall:.4f}")
        print(f"Final Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, preds, target_names=["No CVD","CVD"]))
        
        # 混淆矩陣
        def plot_confusion_matrix(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
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
        plot_confusion_matrix(self.y_test, preds)
        
        return model, self.X_test, self.y_test, raw_probs



    


def plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="blue"):
    """
    繪製所有特徵兩兩組合的錯誤預測散點圖矩陣（正方形比例、點小、字小、避免重疊）
    label_filter: "blue" → 只畫真實標籤為 1；"red" → 只畫真實標籤為 0；"all" → 同時畫
    """


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
    
    # 錯誤樣本細分
    false_neg_idx = X_test.index[(preds == 0) & (y_test == 1)]  # y=1 被預測為 0
    false_pos_idx = X_test.index[(preds == 1) & (y_test == 0)]  # y=0 被預測為 1
    
    X_fn = X_test.loc[false_neg_idx]
    X_fp = X_test.loc[false_pos_idx]
    
    # 要分析的類別特徵
    cat_features = ['active', 'smoke', 'gender', 'alco']
    
    print("\n🔍 錯誤樣本中 y=1 被預測為 0（False Negatives）各特徵的獨特值數量：")
    for col in cat_features:
        print(f"{col}: {X_fn[col].value_counts().to_dict()}")
    
    print("\n🔍 錯誤樣本中 y=0 被預測為 1（False Positives）各特徵的獨特值數量：")
    for col in cat_features:
        print(f"{col}: {X_fp[col].value_counts().to_dict()}")
    
  

def plot_heatmap_for_subset(df, features, title):
    corr = df[features].corr()
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='white')
    plt.title(title, fontsize=13, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_quadrant_heatmaps(X_test, y_test, preds, features):
    """繪製 TP/TN/FP/FN 四象限中指定特徵的 Pearson 關聯性 heatmap"""
    quadrants = {
        "True Positives (y=1, pred=1)": X_test[(preds == 1) & (y_test == 1)],
        "True Negatives (y=0, pred=0)": X_test[(preds == 0) & (y_test == 0)],
        "False Positives (y=0, pred=1)": X_test[(preds == 1) & (y_test == 0)],
        "False Negatives (y=1, pred=0)": X_test[(preds == 0) & (y_test == 1)],
    }

    for title, df in quadrants.items():
        print(f"\n📊 {title} 樣本數量：{len(df)}")
        if len(df) >= 2:  # 至少兩筆資料才能計算相關性
            plot_heatmap_for_subset(df, features, title)
        else:
            print(f"⚠️ {title} 樣本不足，無法繪製 heatmap")

    
 
    
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
    # trainer.tune_knn()
    # trainer.tune_linear_svm()
    # trainer.tune_decision_tree()
    trainer.tune_naive_bayes()
    
    # model, X_test, y_test, threshold, preds = trainer.train_nn()
    # plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="blue")  # 只畫藍點
    # plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="red")   # 只畫紅點
    # plot_error_scatter_matrix(model, X_test, y_test, threshold, label_filter="all")   # 同時畫紅藍點
    # summarize_feature_value_by_true_label_in_errors(model, X_test, y_test, threshold, top_n=10, visualize= False)
    # cat_features = ['active', 'smoke', 'alco', 'gender']
    # plot_confusion_quadrant_heatmaps(X_test, y_test, preds, cat_features )


