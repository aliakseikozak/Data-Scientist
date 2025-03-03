import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_match_outcomes(self, home_score_col='home_score', away_score_col='away_score'):
        outcomes = self.df.apply(lambda row: 'home_win' if row[home_score_col] > row[away_score_col] else
                                             'draw' if row[home_score_col] == row[away_score_col] else
                                             'away_win', axis=1)
        outcome_counts = outcomes.value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
        plt.title('Распределение исходов матчей')
        plt.show()

    def plot_goals_by_year(self, date_col='date', home_score_col='home_score', away_score_col='away_score'):
        self.df['year'] = pd.to_datetime(self.df[date_col]).dt.year
        goals_by_year = self.df.groupby('year')[[home_score_col, away_score_col]].mean()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=goals_by_year, x=goals_by_year.index, y=home_score_col, label='Домашние голы')
        sns.lineplot(data=goals_by_year, x=goals_by_year.index, y=away_score_col, label='Гостевые голы')
        plt.title('Среднее количество голов по годам')
        plt.xlabel('Год')
        plt.ylabel('Среднее количество голов')
        plt.legend()
        plt.show()

    def plot_matches_by_country(self, country_col='country'):
        match_counts = self.df[country_col].value_counts().reset_index()
        match_counts.columns = ['country', 'matches']

        fig = px.choropleth(match_counts, locations='country', locationmode='country names',
                             color='matches', title='Количество матчей по странам')
        fig.show()

    def plot_neutral_fields(self, neutral_col='neutral'):
        neutral_counts = self.df[neutral_col].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=neutral_counts.index, y=neutral_counts.values)
        plt.title('Количество матчей на нейтральных полях')
        plt.xlabel('Нейтральное поле')
        plt.ylabel('Количество матчей')
        plt.xticks([0, 1], ['Нет', 'Да'])
        plt.show()

    def plot_model_accuracy(self, models, accuracies):
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.ylim(0.85, 0.9)  # Установим лимиты по оси Y
        plt.title('Сравнение точности моделей')
        plt.xlabel('Модели')
        plt.ylabel('Точность')
        plt.axhline(y=0.88, color='gray', linestyle='--', label='Средняя точность')
        plt.legend()
        plt.grid(axis='y')
        plt.show()