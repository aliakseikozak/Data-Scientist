import pandas as pd
import numpy as np
import logging

def predict_match_outcome(date, home_team, away_team, home_score, away_score, tournament, city, country, neutral, model, label_encoders, df):
    logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Начало предсказания исхода матча.")
    logger.info(f"Проверяем данные для команды: {home_team}, {away_team}")

    # Рассчитаем данные для домашней команды
    home_team_data = df[(df['home_team'] == home_team) | (df['away_team'] == home_team)]
    away_team_data = df[(df['home_team'] == away_team) | (df['away_team'] == away_team)]

    logger.info(f"Данные для домашней команды: {home_team_data}")
    logger.info(f"Данные для гостевой команды: {away_team_data}")

    # Средние значения голов
    home_team_avg_goals = home_team_data['home_score'].mean()
    away_team_avg_goals = away_team_data['away_score'].mean()

    logger.info(f"Средние голы домашней команды: {home_team_avg_goals}")
    logger.info(f"Средние голы гостевой команды: {away_team_avg_goals}")

    # Условие для отсутствующих данных
    if pd.isna(home_team_avg_goals):
        home_team_avg_goals = df['home_score'].mean()
        logger.warning(f"Нет данных для {home_team}. Используем общее среднее: {home_team_avg_goals}")

    if pd.isna(away_team_avg_goals):
        away_team_avg_goals = df['away_score'].mean()
        logger.warning(f"Нет данных для {away_team}. Используем общее среднее: {away_team_avg_goals}")

    # Рассчитаем скользящие средние за последние 10 матчей
    home_team_avg_goals_10 = home_team_data['home_score'].rolling(window=10, min_periods=1).mean().iloc[-1] if not home_team_data.empty else df['home_score'].mean()
    away_team_avg_goals_10 = away_team_data['away_score'].rolling(window=10, min_periods=1).mean().iloc[-1] if not away_team_data.empty else df['away_score'].mean()

    logger.info(f"Средние голы домашней команды за последние 10 матчей: {home_team_avg_goals_10}")
    logger.info(f"Средние голы гостевой команды за последние 10 матчей: {away_team_avg_goals_10}")

    # Рассчитаем победы и поражения за последние 5 матчей
    if not home_team_data.empty:
        home_team_wins = home_team_data['outcome'].eq('home_win').rolling(window=5, min_periods=1).sum().iloc[-1]
        home_team_losses = home_team_data['outcome'].eq('away_win').rolling(window=5, min_periods=1).sum().iloc[-1]
    else:
        home_team_wins = home_team_losses = 0
        logger.warning(f"Нет данных для расчета побед и поражений домашней команды: {home_team}")

    if not away_team_data.empty:
        away_team_wins = away_team_data['outcome'].eq('away_win').rolling(window=5, min_periods=1).sum().iloc[-1]
        away_team_losses = away_team_data['outcome'].eq('home_win').rolling(window=5, min_periods=1).sum().iloc[-1]
    else:
        away_team_wins = away_team_losses = 0
        logger.warning(f"Нет данных для расчета побед и поражений гостевой команды: {away_team}")

    # Рассчитаем результаты последних 5 матчей между командами
    last_5_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].sort_values(by='date', ascending=False).head(5)

    last_5_results = []
    for _, match in last_5_matches.iterrows():
        if match['home_team'] == home_team:
            last_5_results.append(match['outcome'])
        else:
            if match['outcome'] == 'home_win':
                last_5_results.append('away_win')
            elif match['outcome'] == 'away_win':
                last_5_results.append('home_win')
            else:
                last_5_results.append('draw')

    last_5_home_wins = last_5_results.count('home_win')
    last_5_away_wins = last_5_results.count('away_win')
    last_5_draws = last_5_results.count('draw')

    logger.info(f"Результаты последних 5 матчей между {home_team} и {away_team}: {last_5_results}")
    logger.info(f"Победы домашней команды в последних 5 матчах: {last_5_home_wins}")
    logger.info(f"Победы гостевой команды в последних 5 матчах: {last_5_away_wins}")
    logger.info(f"Ничьи в последних 5 матчах: {last_5_draws}")

    # Преобразуем дату в год, месяц и день
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    day = date.day

    # Создаем DataFrame с новыми данными
    new_data = {
        'home_team': [home_team],
        'away_team': [away_team],
        'tournament': [tournament],
        'city': [city],
        'country': [country],
        'neutral': [neutral],
        'home_team_avg_goals': [home_team_avg_goals],
        'away_team_avg_goals': [away_team_avg_goals],
        'home_team_avg_goals_10': [home_team_avg_goals_10],
        'away_team_avg_goals_10': [away_team_avg_goals_10],
        'home_team_wins': [home_team_wins],
        'home_team_losses': [home_team_losses],
        'away_team_wins': [away_team_wins],
        'away_team_losses': [away_team_losses],
        'last_5_home_wins': [last_5_home_wins],
        'last_5_away_wins': [last_5_away_wins],
        'last_5_draws': [last_5_draws],
        'year': [year],
        'month': [month],
        'day': [day]
    }
    new_df = pd.DataFrame(new_data)
    logger.info(f"Создан новый DataFrame с данными: {new_data}")

    # Закодируем категориальные признаки
    categorical_features = ['home_team', 'away_team', 'tournament', 'city', 'country']
    for feature in categorical_features:
        le = label_encoders[feature]
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')
        unique_values = new_df[feature].unique()
        for value in unique_values:
            if value not in le.classes_:
                logger.warning(f"Найдена новая категория '{value}' для признака '{feature}'. Заменено на 'unknown'.")
                new_df[feature] = new_df[feature].replace(value, 'unknown')
        new_df[feature] = le.transform(new_df[feature])

    # Определим, какие признаки ожидает модель
    try:
        expected_features = model.booster_.feature_name()
    except AttributeError:
        expected_features = model.feature_names_in_

    logger.info(f"Ожидаемые признаки модели: {expected_features}")

    # Удалим лишние признаки, если они не ожидаются моделью
    for feature in ['year', 'month', 'day']:
        if feature not in expected_features:
            new_df = new_df.drop(feature, axis=1)

    # Выберем только те признаки, которые ожидает модель
    X_new = new_df[expected_features]

    # Сделаем предсказание
    prediction = model.predict(X_new)[0]
    logger.info(f"Предсказанное значение: {prediction}")

    # Проверяем тип предсказанного значения
    if isinstance(prediction, (int, np.integer)):
        # Преобразуем предсказанное значение в строку
        if prediction == 0:
            prediction_str = 'away_win'
        elif prediction == 1:
            prediction_str = 'draw'
        elif prediction == 2:
            prediction_str = 'home_win'
        else:
            logger.warning(f"Неизвестное предсказанное значение: {prediction}")
            return "Неизвестный исход матча"
    elif isinstance(prediction, str):
        # Если это текст, используем его напрямую
        prediction_str = prediction
    else:
        logger.warning("Неизвестный тип предсказанного значения.")
        return "Ошибка: Неизвестный тип предсказанного значения."

    # Расшифруем предсказание на русском языке
    outcome_mapping = {
        'home_win': 'Победа домашней команды',
        'draw': 'Ничья',
        'away_win': 'Победа гостевой команды'
    }
    
    return outcome_mapping.get(prediction_str, "Неизвестный исход матча")