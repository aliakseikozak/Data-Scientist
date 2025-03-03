import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def adaboost_model(df):
    # Создаем целевую переменную (исход матча)
    df['outcome'] = df.apply(
        lambda row: 'home_win' if row['home_score'] > row['away_score'] else
                    'draw' if row['home_score'] == row['away_score'] else
                    'away_win', axis=1
    )

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

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Балансировка классов с помощью Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    # Создаем базовый классификатор (слабая модель)
    base_model = RandomForestClassifier(n_estimators=300, random_state=42)

    # Создаем и обучаем модель AdaBoost
    ada_model = AdaBoostClassifier(estimator=base_model, n_estimators=50, random_state=42)  # Изменено на estimator
    ada_model.fit(X_train_balanced, y_train_balanced)

    # Оцениваем модель на тестовых данных
    y_pred = ada_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"AdaBoost Accuracy: {accuracy:.2f}")
    print("AdaBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    return ada_model, label_encoders  # Возвращаем обученную модель и label encoders