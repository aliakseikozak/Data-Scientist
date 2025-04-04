import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf

logger = logging.getLogger(__name__)

def create_goals_distribution(ax, df):
    """График распределения голов."""
    bins = np.arange(-0.5, df['TotalGoals'].max()+1.5, 1)
    ax_obj = sns.histplot(data=df, x='TotalGoals', bins=bins, discrete=True,
                         stat='count', kde=False, color='#4e79a7', ax=ax)
    
    # Добавляем подписи
    for p in ax_obj.patches:
        if p.get_height() > 0:
            ax_obj.annotate(f'{int(p.get_height())}', 
                          (p.get_x() + p.get_width()/2, p.get_height()), 
                          ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Точное распределение голов', pad=15, fontsize=14)
    ax.set_xlabel('Количество голов', fontsize=12)
    ax.set_ylabel('Количество матчей', fontsize=12)
    ax.set_xticks(range(int(df['TotalGoals'].max())+1))
    ax.grid(axis='y', alpha=0.3)

def create_results_pie(ax, df):
    """Круговая диаграмма результатов матчей."""
    ftr_counts = df['FTR'].value_counts()
    colors = ['#2ca02c', '#1f77b4', '#d62728']  # Зеленый, синий, красный
    ftr_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors,
                   labels=['Победа дома', 'Ничья', 'Победа в гостях'],
                   textprops={'fontsize': 12}, ax=ax)
    ax.set_title('Распределение результатов', pad=15, fontsize=14)
    ax.set_ylabel('')

def create_correlation_heatmap(ax, df):
    """Тепловая карта корреляций."""
    corr_data = df[['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']].fillna(0)
    corr_matrix = corr_data.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1, 
        vmax=1,
        fmt=".2f",
        linewidths=.5,
        annot_kws={"size": 9},
        cbar_kws={'label': 'Сила связи'},
        ax=ax
    )
    ax.set_title('Корреляция показателей', pad=15, fontsize=14)

def create_monthly_goals_trend(ax, df):
    """График голов по месяцам."""
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_goals = df.groupby('YearMonth')['TotalGoals'].mean().reset_index()
    
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    all_months = pd.period_range(
        start=f'{min_date.year}-{min_date.month}',
        end=f'{max_date.year}-{max_date.month}',
        freq='M'
    )
    
    monthly_goals_full = pd.DataFrame({'YearMonth': all_months})
    monthly_goals_full = monthly_goals_full.merge(monthly_goals, how='left')
    x_labels = [f"{period.month}\n{period.year}" for period in all_months]
    
    bars = ax.bar(range(len(all_months)), monthly_goals_full['TotalGoals'], 
                 color='#9467bd', width=0.8)
    
    for i, val in enumerate(monthly_goals_full['TotalGoals']):
        if pd.isna(val):
            bars[i].set_color('lightgray')
            bars[i].set_alpha(0.5)
            ax.text(i, 0.1, 'Нет матчей', 
                   ha='center', va='bottom', rotation=90, fontsize=9)
        else:
            ax.text(i, val, f'{val:.1f}', 
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Среднее количество голов по месяцам', pad=15, fontsize=14)
    ax.set_xlabel('Месяц и год', fontsize=12)
    ax.set_ylabel('Средние голы', fontsize=12)
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.grid(axis='y', alpha=0.3)

def create_team_ratings_comparison(ax, df, top_n=20, min_matches=5):
    """Улучшенная визуализация сравнения рейтингов."""
    # 1. Подготовка данных
    home = df.groupby('HomeTeam')['HomeRating'].agg(['mean', 'count'])
    away = df.groupby('AwayTeam')['AwayRating'].agg(['mean', 'count'])
    
    ratings = pd.concat([
        home.rename(columns={'mean': 'HomeRating', 'count': 'HomeMatches'}),
        away.rename(columns={'mean': 'AwayRating', 'count': 'AwayMatches'})
    ], axis=1)
    
    # Фильтрация команд с достаточным количеством матчей
    valid_teams = ratings[
        (ratings['HomeMatches'] >= min_matches) & 
        (ratings['AwayMatches'] >= min_matches)
    ].copy()
    
    valid_teams['Overall'] = (valid_teams['HomeRating'] + valid_teams['AwayRating']) / 2
    top_teams = valid_teams.nlargest(top_n, 'Overall').sort_values('Overall')
    
    # 2. Визуализация
    y = np.arange(len(top_teams))
    height = 0.35
    
    # Цвета из профессиональной палитры
    home_color = '#3498db'  # Синий
    away_color = '#e74c3c'  # Красный
    
    # Столбцы
    ax.barh(y - height/2, top_teams['HomeRating'], height, 
            color=home_color, label='Домашний рейтинг')
    ax.barh(y + height/2, top_teams['AwayRating'], height, 
            color=away_color, label='Гостевой рейтинг')
    
    # Линии среднего
    for i, rating in enumerate(top_teams['Overall']):
        ax.axvline(rating, ymin=(i-height/2)/len(top_teams), 
                  ymax=(i+height/2)/len(top_teams), 
                  color='gray', linestyle=':', alpha=0.7)
    
    # Подписи
    for i, (team, row) in enumerate(top_teams.iterrows()):
        ax.text(row['HomeRating'] + 1, i - height/2, f"{row['HomeRating']:.1f}",
                va='center', color=home_color, fontsize=9)
        ax.text(row['AwayRating'] + 1, i + height/2, f"{row['AwayRating']:.1f}",
                va='center', color=away_color, fontsize=9)
        ax.text(-5, i, team, ha='right', va='center', fontsize=10)
    
    # Настройки графика
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_title(f'Топ-{top_n} команд по рейтингам (минимум {min_matches} матчей дома/в гостях)', 
                pad=15, fontsize=12)
    ax.set_xlabel('Рейтинг (0-100)', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.2)
    ax.set_xlim(left=-10, right=110)
    
    # Информация о данных
    ax.text(0.95, 0.95, 
            f"На основе {len(df)} матчей\n{len(valid_teams)} команд с ≥{min_matches} матчами",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round')) 

def perform_eda(df, save_dir='visualizations'):
    """
    Выполняет EDA анализ с отображением и сохранением графиков.
    
    Args:
        df: DataFrame с данными
        save_dir: Директория для сохранения графиков
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Проверка данных
    logger.info("Уникальные значения голов: %s", sorted(df['TotalGoals'].unique()))
    odd_goals = [x for x in df['TotalGoals'].unique() if x % 2 != 0]
    logger.info("Наличие нечётных значений: %s", "Есть" if odd_goals else "Нет")

    # 1. Распределение голов
    plt.figure(figsize=(10, 6))
    create_goals_distribution(plt.gca(), df)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goals_distribution.png'), dpi=120, bbox_inches='tight')
    plt.show()
    
    # 2. Круговая диаграмма результатов
    plt.figure(figsize=(10, 6))
    create_results_pie(plt.gca(), df)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_pie.png'), dpi=120, bbox_inches='tight')
    plt.show()
    
    # 3. Тепловая карта корреляций
    plt.figure(figsize=(12, 8))
    create_correlation_heatmap(plt.gca(), df)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.show()
    
    # 4. Тренд голов по месяцам
    plt.figure(figsize=(14, 6))
    create_monthly_goals_trend(plt.gca(), df)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'monthly_goals_trend.png'), dpi=120, bbox_inches='tight')
    plt.show()
    
    # 5. Сравнение рейтингов команд
    plt.figure(figsize=(12, 8))
    create_team_ratings_comparison(plt.gca(), df)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'team_ratings_comparison.png'), dpi=120, bbox_inches='tight')
    plt.show()
    
    logger.info("Все графики сохранены в директорию %s и отображены на экране", save_dir)


def plot_model_comparison(models_comparison: pd.DataFrame) -> plt.Figure:
    """
    Сравнение моделей по метрикам MSE и MAE
    
    Args:
        models_comparison: DataFrame с результатами сравнения моделей
        
    Returns:
        Figure: Объект графика
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График MSE
    sns.barplot(data=models_comparison, x='Model', y='MSE', hue='Target', ax=ax1)
    ax1.set_title('Сравнение MSE моделей')
    ax1.set_xticks(rotation=45)
    ax1.set_ylabel('MSE')
    
    # График MAE
    sns.barplot(data=models_comparison, x='Model', y='MAE', hue='Target', ax=ax2)
    ax2.set_title('Сравнение MAE моделей')
    ax2.set_xticks(rotation=45)
    ax2.set_ylabel('MAE')
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    title: str
) -> plt.Figure:
    """
    График фактических vs предсказанных значений
    
    Args:
        y_test: Фактические значения
        y_pred: Предсказанные значения
        title: Заголовок графика
        
    Returns:
        Figure: Объект графика
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_title(f'{title} (R2={r2_score(y_test, y_pred):.2f})')
    ax.set_xlabel('Фактические значения')
    ax.set_ylabel('Предсказанные значения')
    return fig

def plot_residuals_analysis(
    y_test: np.ndarray, 
    y_pred: np.ndarray,
    title: str = ''
) -> Tuple[plt.Figure, dict]:
    """
    Анализ остатков модели
    
    Args:
        y_test: Фактические значения
        y_pred: Предсказанные значения
        title: Заголовок графика
        
    Returns:
        Tuple: (Figure, статистика ошибок)
    """
    residuals = y_test - y_pred
    stats = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'median': np.median(residuals),
        'min': residuals.min(),
        'max': residuals.max()
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Гистограмма остатков
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title(f'Распределение ошибок {title}')
    ax1.set_xlabel('Ошибка (Факт - Прогноз)')
    
    # Остатки vs предсказанные
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(f'Остатки vs Предсказанные {title}')
    ax2.set_xlabel('Предсказанные значения')
    ax2.set_ylabel('Остатки')
    
    plt.tight_layout()
    return fig, stats

def plot_feature_importance(
    model, 
    feature_names: list, 
    top_n: int = 20,
    title: str = 'Важность признаков'
) -> plt.Figure:
    """
    Визуализация важности признаков
    
    Args:
        model: Обученная модель (должен иметь feature_importances_)
        feature_names: Список названий признаков
        top_n: Количество топовых признаков для отображения
        title: Заголовок графика
        
    Returns:
        Figure: Объект графика
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Важность признака')
    ax.set_title(title)
    return fig

def save_plot(fig: plt.Figure, filename: str, dpi: int = 120) -> None:
    """
    Сохранение графика в файл
    
    Args:
        fig: Объект графика
        filename: Имя файла для сохранения
        dpi: Качество изображения
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_best_model_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot с цветом по плотности точек
    sns.regplot(x=y_true, y=y_pred, 
                scatter_kws={'alpha': 0.4, 'color': 'royalblue'},
                line_kws={'color': 'red', 'linestyle': '--', 'label': 'Идеальная линия'})
    
    # Создаем заголовок
    title = f'{model_name}\nR²: {r2_score(y_true, y_pred):.3f} | MAE: {mean_absolute_error(y_true, y_pred):.3f}'
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Фактические значения', fontsize=12)
    plt.ylabel('Предсказанные значения', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2)
    
    # Генерация имени файла из заголовка
    filename = (
        title.replace('\n', ' ')  # Заменяем переносы строк на пробелы
             .replace(':', '-')   # Заменяем двоеточия
             .replace('|', '-')   # Заменяем другие спецсимволы
             .replace('  ', ' ')  # Убираем двойные пробелы
             .strip() + '.png'    # Добавляем расширение
    )
    
    # Создаем папку для сохранения, если ее нет
    os.makedirs('visualizations', exist_ok=True)
    
    # Сохраняем с автоматическим именем
    plt.savefig(f'visualizations/{filename}', bbox_inches='tight', dpi=300)
    plt.show()


def compare_models_metrics(y_true, models_predictions):
    metrics = []
    for name, pred in models_predictions.items():
        metrics.append({
            'Model': name,
            'MAE': mean_absolute_error(y_true, pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
            'R²': r2_score(y_true, pred)
        })
    
    df = pd.DataFrame(metrics).set_index('Model')
    
    # График
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', subplots=True, layout=(1, 3), 
            figsize=(14, 5), sharex=True, legend=False)
    plt.suptitle('Сравнение моделей', y=1.05)
    plt.tight_layout()
    plt.savefig('visualizations/metric_comparison.png')
    plt.show()
    
    return df    

def plot_residuals_distribution(y_true, models_predictions):
    plt.figure(figsize=(12, 6))
    
    for name, pred in models_predictions.items():
        residuals = y_true - pred
        sns.kdeplot(residuals, label=name, alpha=0.7, linewidth=2)
    
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Распределение ошибок по моделям', fontsize=14)
    plt.xlabel('Ошибка (Факт - Прогноз)', fontsize=12)
    plt.ylabel('Плотность', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig('visualizations/distribution_errors_across_models.png')
    plt.show()    
###########################################################################
def plot_goals_timeseries(df):
    if 'Date' not in df.columns:
        print("Нет данных о дате для построения временного ряда")
        return
    
    # Создаем копию данных для группировки
    df_plot = df.copy()
    df_plot['Date'] = pd.to_datetime(df_plot['Date'])
    
    # Группировка по месяцам (можно изменить на недели/кварталы)
    df_plot = df_plot.set_index('Date')
    monthly = df_plot.resample('M').agg({
        'FTHG': ['mean', 'count'],
        'FTAG': ['mean', 'count']
    })
    
    # Фильтруем периоды с малым количеством матчей
    monthly = monthly[monthly[('FTHG', 'count')] > 3]  # Только месяцы с >3 матчами
    
    plt.figure(figsize=(14, 7))
    
    # Скользящее среднее
    monthly[('FTHG', 'mean')].rolling(3).mean().plot(
        color='blue', linewidth=2, 
        label='Домашние голы (среднее за 3 мес)')
    monthly[('FTAG', 'mean')].rolling(3).mean().plot(
        color='red', linewidth=2,
        label='Гостевые голы (среднее за 3 мес)')
    
    # Точечные значения
    plt.scatter(monthly.index, monthly[('FTHG', 'mean')], 
               color='blue', alpha=0.5, label='Домашние голы (месячное среднее)')
    plt.scatter(monthly.index, monthly[('FTAG', 'mean')],
               color='red', alpha=0.5, label='Гостевые голы (месячное среднее)')
    
    plt.title('Динамика голов по времени (группировка по месяцам)', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Среднее количество голов', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Форматирование дат
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()  

def plot_team_form(df, team_name):
    if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
        print("Нет данных о командах")
        return
    
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    team_matches['Date'] = pd.to_datetime(team_matches['Date'])
    team_matches = team_matches.sort_values('Date')
    
    # Создаем признаки формы
    team_matches['TeamGoals'] = np.where(team_matches['HomeTeam'] == team_name, 
                                        team_matches['FTHG'], team_matches['FTAG'])
    team_matches['Conceded'] = np.where(team_matches['HomeTeam'] == team_name, 
                                      team_matches['FTAG'], team_matches['FTHG'])
    team_matches['Result'] = np.where(team_matches['HomeTeam'] == team_name, 
                                    team_matches['FTR'].map({'H': 1, 'D': 0.5, 'A': 0}),
                                    team_matches['FTR'].map({'A': 1, 'D': 0.5, 'H': 0}))
    
    # Скользящие средние
    for window in [3, 5, 10]:
        team_matches[f'Goals_MA_{window}'] = team_matches['TeamGoals'].rolling(window=window).mean()
        team_matches[f'Form_MA_{window}'] = team_matches['Result'].rolling(window=window).mean()
    
    plt.figure(figsize=(15, 8))
    
    # График голов
    plt.subplot(2, 1, 1)
    plt.plot(team_matches['Date'], team_matches['TeamGoals'], 'o-', label='Голы за матч')
    for window in [3, 5, 10]:
        plt.plot(team_matches['Date'], team_matches[f'Goals_MA_{window}'], 
                label=f'Среднее за {window} матчей')
    plt.title(f'Голы команды {team_name} по времени')
    plt.legend()
    plt.grid(True)
    
    # График формы
    plt.subplot(2, 1, 2)
    plt.plot(team_matches['Date'], team_matches['Result'], 'o-', label='Результат за матч')
    for window in [3, 5, 10]:
        plt.plot(team_matches['Date'], team_matches[f'Form_MA_{window}'], 
                label=f'Форма за {window} матчей')
    plt.title(f'Форма команды {team_name} по времени')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()  

def plot_matches_over_time(df):
    if 'Date' not in df.columns:
        print("Нет данных о дате для анализа")
        return
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])  # Убедимся, что даты в правильном формате
    
    # Добавим новый столбец для месяца и года
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Подсчитаем количество матчей по месяцам
    matches_per_month = df.groupby('YearMonth').size()
    
    plt.figure(figsize=(14, 7))
    matches_per_month.plot(kind='bar', color='skyblue', alpha=0.7)
    
    plt.title('Количество матчей по месяцам', fontsize=16)
    plt.xlabel('Год и месяц', fontsize=14)
    plt.ylabel('Количество матчей', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()         

def plot_goals_and_shots_over_time(df, date_column='Date', resample_freq='M'):
    """
    Строит график, показывающий количество голов и ударов в створ ворот по датам.
    
    :param df: DataFrame, содержащий данные о матчах.
    :param date_column: Столбец с датами (по умолчанию 'Date').
    :param resample_freq: Частота для усреднения данных (по умолчанию 'М' для месяцев).
    """
    # Проверка наличия необходимых столбцов
    required_columns = ['FTHG', 'FTAG', 'HS', 'AS', date_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Отсутствуют следующие необходимые столбцы: {missing_columns}")
        return

    # Убедитесь, что даты правильно распознаны
    df[date_column] = pd.to_datetime(df[date_column])

    # Создаем новые столбцы для анализа
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['TotalShots'] = df['HS'] + df['AS']

    # Группируем данные по дате, усредняя по заданной частоте
    daily_data = df.groupby(date_column).agg({'TotalGoals': 'sum', 'TotalShots': 'sum'}).reset_index()
    daily_data.set_index(date_column, inplace=True)
    
    # Усреднение по указанной частоте
    resampled_data = daily_data.resample(resample_freq).sum().reset_index()

    # Построение графика
    plt.figure(figsize=(16, 8))
    plt.plot(resampled_data[date_column], resampled_data['TotalGoals'], marker='o', label='Общее количество голов', color='blue', alpha=0.6)
    plt.plot(resampled_data[date_column], resampled_data['TotalShots'], marker='o', label='Общее количество ударов в створ', color='orange', alpha=0.6)

    plt.title('Количество голов и ударов в створ по датам', fontsize=20)
    plt.xlabel('Дата', fontsize=16)
    plt.ylabel('Количество', fontsize=16)
    
    # Настройка меток на оси X для отображения года и месяца
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Измените интервал, если нужно
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


