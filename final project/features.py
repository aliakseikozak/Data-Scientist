import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def standardize_dataset(df):
    """Приводит датасет к стандартному формату"""
    standardized = df.copy()
    
    # 1. Переименовываем столбцы для унификации
    column_mapping = {
        'IWH': 'BFH', 'IWD': 'BFD', 'IWA': 'BFA',
        'IWCH': 'BFCH', 'IWCD': 'BFCD', 'IWCA': 'BFCA',
        'VCH': '1XBH', 'VCD': '1XBD', 'VCA': '1XBA',
        'VCCH': '1XBCH', 'VCCD': '1XBCD', 'VCCA': '1XBCA'
    }
    standardized = standardized.rename(columns=column_mapping)
    
    # 2. Добавляем отсутствующие столбцы с NaN значениями
    expected_columns = [
        'BFH', 'BFD', 'BFA', 'BFEH', 'BFED', 'BFEA', 
        'BFCH', 'BFCD', 'BFCA', 'BFECH', 'BFECD', 'BFECA',
        '1XBH', '1XBD', '1XBA', '1XBCH', '1XBCD', '1XBCA',
        'BFE>2.5', 'BFE<2.5', 'BFEC>2.5', 'BFEC<2.5',
        'BFEAHH', 'BFEAHA', 'BFECAHH', 'BFECAHA'
    ]
    
    for col in expected_columns:
        if col not in standardized.columns:
            standardized[col] = np.nan
    
    # 3. Удаляем лишние столбцы
    columns_to_drop = [col for col in standardized.columns 
                      if col.startswith('IW') or col.startswith('VC')]
    standardized = standardized.drop(columns=columns_to_drop, errors='ignore')
    
    return standardized

def merge_seasons(input_folder='data', output_file='season_merged.csv'):
    """
    Объединяет все файлы сезонов в один стандартизированный файл
    
    Args:
        input_folder (str): Папка с файлами сезонов
        output_file (str): Имя выходного файла
        
    Returns:
        pd.DataFrame: Объединенный датасет
    """
    merged_data = pd.DataFrame()
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            print(f"Обработка файла: {filename}")
            
            season_data = pd.read_csv(filepath)
            standardized_data = standardize_dataset(season_data)
            merged_data = pd.concat([merged_data, standardized_data], ignore_index=True)
    
    # Сохраняем результат
    Path('processed_data').mkdir(exist_ok=True)
    output_path = os.path.join('processed_data', output_file)
    merged_data.to_csv(output_path, index=False)
    print(f"\nОбъединенные данные сохранены в {output_path}")
    
    return merged_data

def preprocess_data(df):

    """Базовая предобработка с проверкой структуры данных"""
    # Проверяем наличие обязательных столбцов
    required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Обязательный столбец {col} отсутствует в данных")
    
    # Стандартизируем формат если нужно
    if 'IWH' in df.columns:
        df = standardize_dataset(df)
            
    """Базовая предобработка: даты, пропуски, кодирование."""
    # 1. Первичная проверка пропущенных значений
    initial_missing = df.isnull().sum()
    if initial_missing.any():
        print("\n[ПЕРВИЧНАЯ ПРОВЕРКА] Обнаружены пропущенные значения:")
        print(initial_missing[initial_missing > 0])
    
    # 2. Обработка букмекерских коэффициентов
    bk_cols = ['BWH', 'BWD', 'BWA', 'BFH', 'BFD', 'BFA', 
               'BWCH', 'BWCD', 'BWCA', '1XBH', '1XBD', '1XBA',
               'P>2.5', 'P<2.5', 'BFE>2.5', 'BFE<2.5', 'PC>2.5', 'PC<2.5']
    
    for col in bk_cols:
        if col in df.columns and df[col].isnull().any():
            if col.startswith(('BW', 'BF', '1XB')):
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"Заполнены пропуски в {col} медианным значением: {fill_value:.2f}")
            elif col.startswith('P'):
                df[col] = df[col].fillna(0.5)
                print(f"Заполнены пропуски в {col} нейтральным значением 0.5")
            elif col.startswith('PC'):
                df[col] = df[col].fillna(2.5)
                print(f"Заполнены пропуски в {col} средним тоталом 2.5")
    
    # 3. Обработка даты
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    invalid_dates = df['Date'].isna()
    if invalid_dates.any():
        print(f"\nОбнаружено {invalid_dates.sum()} невалидных дат. Они будут исключены из временных признаков.")
    
    # 4. Заполнение остальных числовых колонок
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(bk_cols)
    for col in numeric_cols:
        if df[col].isnull().any():
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"Заполнены пропуски в {col} медианным значением: {fill_val:.2f}")
    
    # 5. Заполнение категориальных колонок
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            print(f"Заполнены пропуски в категориальной колонке {col} значением 'Unknown'")
    
    # 6. Финальная проверка пропусков (добавлено)
    final_missing = df.isnull().sum()
    remaining_missing = final_missing[final_missing > 0]
    
    if not remaining_missing.empty:
        print("\n[ФИНАЛЬНАЯ ПРОВЕРКА] Остались необработанные пропуски:")
        print(remaining_missing)
    else:
        print("\n[ФИНАЛЬНАЯ ПРОВЕРКА] Все пропуски успешно обработаны!")
    
    # 7. Добавление временных признаков
    valid_dates = df['Date'].notna()
    df.loc[valid_dates, 'Year'] = df.loc[valid_dates, 'Date'].dt.year
    df.loc[valid_dates, 'Month'] = df.loc[valid_dates, 'Date'].dt.month
    df.loc[valid_dates, 'DayOfWeek'] = df.loc[valid_dates, 'Date'].dt.dayofweek
    
    # 8. Кодирование категорий
    le = LabelEncoder()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
    le.fit(teams)
    df['HomeTeam_encoded'] = df['HomeTeam'].map(lambda x: le.transform([x])[0] if pd.notna(x) else -1)
    df['AwayTeam_encoded'] = df['AwayTeam'].map(lambda x: le.transform([x])[0] if pd.notna(x) else -1)
    
    # 9. Производные признаки
    df['TotalGoals'] = df['FTHG'].fillna(0) + df['FTAG'].fillna(0)
    df['GoalDiff'] = df['FTHG'].fillna(0) - df['FTAG'].fillna(0)
    
    print("\nПредобработка данных завершена успешно!")
    return df

def enrich_data(df):
    """Генерация сложных аналитических признаков с раздельными рейтингами."""
    # Сортируем по дате, чтобы не было утечки будущего
    df = df.sort_values('Date').copy()
    
    # Рассчитываем рейтинги (без утечки данных)
    team_ratings = _calculate_team_ratings(df)
    
    # Применяем рейтинги
    df['HomeRating'] = df['HomeTeam'].map(lambda x: team_ratings.get(x, {}).get('HomeRating', 50))
    df['AwayRating'] = df['AwayTeam'].map(lambda x: team_ratings.get(x, {}).get('AwayRating', 50))
    df['RatingDiff'] = df['HomeRating'] - df['AwayRating']
    
    # Форма команд (последние 5 матчей)
    for team in df['HomeTeam'].unique():
        df.loc[df['HomeTeam'] == team, 'HomeForm'] = _calculate_form(df, team, is_home=True)
        df.loc[df['AwayTeam'] == team, 'AwayForm'] = _calculate_form(df, team, is_home=False)
    
    # История встреч (только предыдущие матчи между этими командами)
    df = _add_h2h_features(df)

    # Разница форм команд (относительная форма)
    df['Относительная_форма'] = df['HomeForm'] - df['AwayForm']
    
    # Разница коэффициентов на победу
    df['Коэф_разница'] = df['BWA'] - df['BWH']
    
    # Правильный расчет силы атаки на основе исторических данных
    df['HomeAttackStrength'] = df.apply(lambda row: _calculate_attack_strength(row['HomeTeam'], row['Date'], df, is_home=True), axis=1)
    df['AwayAttackStrength'] = df.apply(lambda row: _calculate_attack_strength(row['AwayTeam'], row['Date'], df, is_home=False), axis=1)
    df['Сила_атаки_разница'] = df['HomeAttackStrength'] - df['AwayAttackStrength']

    # Добавляем новые признаки
    df['AttackDefenseBalance'] = df['HomeAttackStrength'] * (1 - df['AwayAttackStrength'])  # Исправлено: используем AwayAttackStrength как прокси для защиты
    
    # Улучшенный расчет силы атаки
    df['EnhancedHomeAttack'] = df.apply(lambda row: _enhanced_attack_strength(row['HomeTeam'], row['Date'], df), axis=1)
    df['EnhancedAwayAttack'] = df.apply(lambda row: _enhanced_attack_strength(row['AwayTeam'], row['Date'], df), axis=1)
    
    # Относительная сила команд
    df['RelativeStrength'] = df['HomeRating'] / (df['AwayRating'] + 1e-6)
    
    # Тренд формы
    df['HomeTrend'] = df.apply(lambda row: _calculate_trend(row['HomeTeam'], row['Date'], df, is_home=True), axis=1)
    df['AwayTrend'] = df.apply(lambda row: _calculate_trend(row['AwayTeam'], row['Date'], df, is_home=False), axis=1)

    df['AvgGoalDiff'] = df['HomeForm'] - df['AwayForm']
    df['BookmakerImbalance'] = df['BWH'] / (df['BWA'] + 1e-6)
    
    return df

def _calculate_team_ratings(df):
    """Расчет рейтингов команд с учетом силы соперников и ключевых метрик."""
    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательный столбец: {col}")

    # Сортируем по дате для правильного расчета
    df = df.sort_values('Date').copy()
    
    # --- 1. Базовые статистики ---
    home_stats = df.groupby('HomeTeam').agg({
        'FTHG': ['mean', 'count'],  # Средние голы дома + кол-во матчей
        'FTAG': 'mean',              # Средние пропущенные дома
        'FTR': lambda x: (x == 'H').mean()  # Процент побед дома
    })
    home_stats.columns = ['HomeGoals_mean', 'HomeGoals_count', 'ConcededHome_mean', 'HomeWinRate']

    away_stats = df.groupby('AwayTeam').agg({
        'FTAG': ['mean', 'count'],  # Средние голы в гостях + кол-во матчей
        'FTHG': 'mean',              # Средние пропущенные в гостях
        'FTR': lambda x: (x == 'A').mean()  # Процент побед в гостях
    })
    away_stats.columns = ['AwayGoals_mean', 'AwayGoals_count', 'ConcededAway_mean', 'AwayWinRate']

    # Объединяем статистики
    stats = home_stats.join(away_stats, how='outer')

    # --- 2. Учет силы соперников (упрощенный Elo) ---
    def _calculate_strength(team, is_home=True):
        """Рассчитывает силу команды с учетом соперников."""
        if is_home:
            games = df[df['HomeTeam'] == team]
            opponent_col = 'AwayTeam'
            result_weight = (games['FTR'] == 'H').astype(int) + (games['FTR'] == 'D').astype(int) * 0.5
        else:
            games = df[df['AwayTeam'] == team]
            opponent_col = 'HomeTeam'
            result_weight = (games['FTR'] == 'A').astype(int) + (games['FTR'] == 'D').astype(int) * 0.5
        
        if len(games) == 0:
            return 50
        
        # Средняя сила соперников (используем их домашний/гостевой рейтинг)
        opponents = games[opponent_col].unique()
        opponent_strength = stats.loc[opponents, 'HomeGoals_mean' if not is_home else 'AwayGoals_mean'].mean()
        
        return (result_weight.mean() * opponent_strength * 100) / (opponent_strength.mean() or 1)

    stats['HomeStrength'] = stats.index.map(lambda x: _calculate_strength(x, is_home=True))
    stats['AwayStrength'] = stats.index.map(lambda x: _calculate_strength(x, is_home=False))

    # --- 3. Нормализация и финальный рейтинг ---
    def _normalize(s):
        return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() > s.min() else 50

    stats['HomeRating'] = (
        0.5 * _normalize(stats['HomeStrength']) +
        0.3 * _normalize(stats['HomeGoals_mean']) +
        0.2 * _normalize(1 / (stats['ConcededHome_mean'] + 0.1))
    ).round(1)

    stats['AwayRating'] = (
        0.5 * _normalize(stats['AwayStrength']) +
        0.3 * _normalize(stats['AwayGoals_mean']) +
        0.2 * _normalize(1 / (stats['ConcededAway_mean'] + 0.1))
    ).round(1)

    # --- 4. Результат ---
    ratings = stats[['HomeRating', 'AwayRating']]
    print("\nИтоговые рейтинги:")
    print(ratings.sort_values('HomeRating', ascending=False).head(10))
    
    return ratings.to_dict('index')    

def _calculate_attack_strength(team, current_date, df, is_home=True, n_matches=10):
    """Рассчитывает силу атаки на основе последних n_matches матчей."""
    # Берем только матчи ДО текущей даты
    past_matches = df[(df['Date'] < current_date) & 
                     ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))]
    
    if len(past_matches) == 0:
        return 0.0  # нейтральное значение
    
    # Берем последние n_matches матчей
    past_matches = past_matches.sort_values('Date', ascending=False).head(n_matches)
    
    if is_home:
        # Для домашней атаки считаем голы дома
        home_matches = past_matches[past_matches['HomeTeam'] == team]
        if len(home_matches) > 0:
            return home_matches['FTHG'].mean()
    else:
        # Для гостевой атаки считаем голы в гостях
        away_matches = past_matches[past_matches['AwayTeam'] == team]
        if len(away_matches) > 0:
            return away_matches['FTAG'].mean()
    
    return 0.0

def _calculate_form(df, team, is_home, n_matches=5):
    """Приватная функция для расчета формы (без утечки данных)."""
    current_date = df['Date'].max() if 'Date' in df.columns else None
    
    # Фильтруем матчи только до текущей даты
    if current_date is not None:
        team_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                         (df['Date'] < current_date)]
    else:
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    
    # Берем последние n_matches матчей
    team_matches = team_matches.sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return 0.0
    
    if is_home:
        # Для домашней формы считаем только домашние матчи
        home_matches = team_matches[team_matches['HomeTeam'] == team]
        if len(home_matches) > 0:
            return (home_matches['FTHG'].mean() - home_matches['FTAG'].mean()) * 1.2
    else:
        # Для гостевой формы считаем только гостевые матчи
        away_matches = team_matches[team_matches['AwayTeam'] == team]
        if len(away_matches) > 0:
            return (away_matches['FTAG'].mean() - away_matches['FTHG'].mean()) * 0.8
    
    return 0.0

def _add_h2h_features(df, n_matches=5):
    """Приватная функция для добавления истории встреч (без утечки данных)."""
    # Создаем копию DataFrame для безопасного добавления признаков
    result_df = df.copy()
    
    # Инициализируем новые колонки
    result_df['H2H_HomeGoals'] = np.nan
    result_df['H2H_AwayGoals'] = np.nan
    result_df['H2H_WinRate'] = np.nan
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        current_date = row['Date']
        
        # Находим предыдущие матчи между этими командами ДО текущей даты
        h2h_matches = df[
            (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            (((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
            (df['Date'] < current_date))
        ].sort_values('Date', ascending=False).head(n_matches)
        
        if len(h2h_matches) > 0:
            # Средние голы домашней команды в этих матчах
            home_goals = []
            away_goals = []
            wins = 0
            
            for _, match in h2h_matches.iterrows():
                if match['HomeTeam'] == home_team:
                    home_goals.append(match['FTHG'])
                    away_goals.append(match['FTAG'])
                    if match['FTR'] == 'H':
                        wins += 1
                else:
                    home_goals.append(match['FTAG'])
                    away_goals.append(match['FTHG'])
                    if match['FTR'] == 'A':
                        wins += 1
            
            result_df.loc[idx, 'H2H_HomeGoals'] = np.mean(home_goals) if home_goals else df['FTHG'].mean()
            result_df.loc[idx, 'H2H_AwayGoals'] = np.mean(away_goals) if away_goals else df['FTAG'].mean()
            result_df.loc[idx, 'H2H_WinRate'] = wins / len(h2h_matches)
        else:
            # Если нет истории встреч, используем средние значения
            result_df.loc[idx, 'H2H_HomeGoals'] = df['FTHG'].mean()
            result_df.loc[idx, 'H2H_AwayGoals'] = df['FTAG'].mean()
            result_df.loc[idx, 'H2H_WinRate'] = 0.33  # Базовый уровень
    
    # Заполняем оставшиеся пропуски средними значениями
    result_df['H2H_HomeGoals'] = result_df['H2H_HomeGoals'].fillna(df['FTHG'].mean())
    result_df['H2H_AwayGoals'] = result_df['H2H_AwayGoals'].fillna(df['FTAG'].mean())
    result_df['H2H_WinRate'] = result_df['H2H_WinRate'].fillna(0.33)
    
    return result_df

def _enhanced_attack_strength(team, date, df, n_matches=10):
    """Улучшенный расчет силы атаки с учетом формы"""
    past_matches = df[(df['Date'] < date) & 
                     ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))]
    past_matches = past_matches.sort_values('Date', ascending=False).head(n_matches)
    
    if len(past_matches) == 0:
        return 0.5
    
    home_attack = past_matches[past_matches['HomeTeam'] == team]['FTHG'].mean()
    away_attack = past_matches[past_matches['AwayTeam'] == team]['FTAG'].mean()
    
    return (home_attack * 0.6 + away_attack * 0.4)  # Больший вес домашним матчам

def _calculate_trend(team, date, df, is_home=True, n_matches=5):
    """Расчет тренда формы (улучшение/ухудшение)"""
    past_matches = df[(df['Date'] < date) & 
                     ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))]
    past_matches = past_matches.sort_values('Date').tail(n_matches * 2)
    
    if len(past_matches) < 3:
        return 0
    
    # Разделяем на ранние и поздние матчи
    early = past_matches.head(n_matches)
    late = past_matches.tail(n_matches)
    
    # Сравниваем производительность
    if is_home:
        early_perf = (early['FTR'] == 'H').mean() 
        late_perf = (late['FTR'] == 'H').mean()
    else:
        early_perf = (early['FTR'] == 'A').mean()
        late_perf = (late['FTR'] == 'A').mean()
    
    return late_perf - early_perf    