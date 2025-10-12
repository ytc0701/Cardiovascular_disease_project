from data_washing import load_and_preprocess
from training import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, X, name):

    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print(f"\n📊 {name} Feature Importance:")
    print(feature_importance)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][:], feature_importance['Importance'][:])
    plt.gca().invert_yaxis()
    plt.title(f'{name} Feature Importance')
    plt.xlabel('Importance')
    plt.show()

if __name__ == "__main__":
    X, y = load_and_preprocess()
    X = X.drop(['male','female','gluc_nor','gluc_abnor','gluc_wellabnor','smoke','alco','active','BMI_normal','BMI_obese'],axis=1)
    trainer = ModelTrainer(X, y)
    trainer.train_random_forest()
    # trainer.train_xgboost()
    # trainer.train_lightgbm()
    # trainer.train_catboost()
    plot_feature_importance(trainer.train_random_forest(), trainer.X_train, "Random Forest")
    # plot_feature_importance(trainer.train_xgboost(), trainer.X_train, "XGBoost")
    # plot_feature_importance(trainer.train_lightgbm(), trainer.X_train, "LightGBM")
    # plot_feature_importance(trainer.train_catboost(), trainer.X_train, "CatBoost")