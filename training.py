from data_washing import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

# XGBoost / LightGBM / CatBoost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Neural Network
from sklearn.neural_network import MLPClassifier


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=40
        )

    def evaluate_model(self, model, name):
        """統一的訓練 + 預測 + 評估流程"""
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, preds))
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


if __name__ == "__main__":
    X, y = load_and_preprocess()

    trainer = ModelTrainer(X, y)

    # 逐一訓練模型
    trainer.train_logistic()
    trainer.train_linear_svm()
    trainer.train_decision_tree()
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.train_lightgbm()
    trainer.train_catboost()
    trainer.train_nn()
    trainer.train_knn(5)
    trainer.train_naive_bayes()