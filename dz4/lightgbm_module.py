import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def lightgbm_train_model(df):
    # Преобразование целевой переменной
    df['outcome'] = df.apply(
        lambda row: 'home_win' if row['home_score'] > row['away_score'] else
                    'away_win' if row['home_score'] < row['away_score'] else
                    'draw', axis=1
    )

    # Преобразование столбца даты
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Удаляем исходный столбец даты (теперь он больше не нужен)
    df = df.drop(['date'], axis=1)

    # Кодируем категориальные признаки
    label_encoders = {}
    categorical_features = ['home_team', 'away_team', 'tournament', 'city', 'country']
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

    # Разделяем данные на признаки и целевую переменную
    X = df[['home_team', 'away_team', 'tournament', 'city', 'country', 'neutral', 
            'home_team_wins', 'home_team_losses', 'away_team_wins', 'away_team_losses',
            'home_team_avg_goals', 'away_team_avg_goals', 'home_team_avg_goals_10', 'away_team_avg_goals_10',
            'last_5_home_wins', 'last_5_away_wins', 'last_5_draws']]
    y = df['outcome']

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создаем и обучаем модель LightGBM
    model = lgb.LGBMClassifier(num_leaves=31, min_child_samples=20, learning_rate=0.1, n_estimators=100)
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('LightGBM Classification Report:')
    print(classification_report(y_test, y_pred))

    return model, label_encoders