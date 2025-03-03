import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def xgboost_model(df):
    # Создаем целевую переменную (исход матча)

    # Кодируем категориальные признаки
    categorical_features = ['home_team', 'away_team', 'tournament', 'city', 'country']
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

    # Разделяем данные на признаки (X) и целевую переменную (y)
    X = df[['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral', 
            'home_team_wins', 'home_team_losses', 'away_team_wins', 'away_team_losses',
            'home_team_avg_goals', 'away_team_avg_goals', 'home_team_avg_goals_10', 'away_team_avg_goals_10',
            'last_5_home_wins', 'last_5_away_wins', 'last_5_draws']]
    y = df['outcome']

    # Кодируем целевую переменную
    le_outcome = LabelEncoder()
    y_encoded = le_outcome.fit_transform(y)
    label_encoders['outcome'] = le_outcome  # Сохраняем кодировщик для целевой переменной

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Балансировка классов с помощью Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    # Создаем и обучаем модель XGBoost
    xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train_balanced, y_train_balanced)

    # Оцениваем модель на тестовых данных
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.2f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_outcome.classes_))  # Используем названия классов для отчета

    return xgb_model, label_encoders  # Возвращаем обученную модель и label encoders