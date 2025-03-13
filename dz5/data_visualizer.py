import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_average_price_vs_mileage_audi(self):
        # Фильтрация DataFrame для бренда Audi
        audi_df = self.df[self.df['Brand'] == 'Audi']
        bins = range(0, 300001, 20000) 
        labels = [f'{i}-{i+20000}' for i in range(0, 300000, 20000)]
        audi_df['MileageRange'] = pd.cut(audi_df['Mileage'], bins=bins, labels=labels, right=False)
        average_price = audi_df.groupby('MileageRange')['Price'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=average_price, x='MileageRange', y='Price', marker='o')
        plt.title('Средняя цена автомобилей Audi в зависимости от пробега')
        plt.xlabel('Пробег (интервалы по 20 000 км)')
        plt.ylabel('Средняя цена')
        plt.grid()
        plt.xticks(rotation=45)  
        plt.show()

    def plot_average_price_by_fuel_type(self):
        plt.figure(figsize=(8, 5))
        sns.barplot(data=self.df, x='Fuel_Type', y='Price', estimator='mean')
        plt.title('Средняя цена по типу топлива')
        plt.xlabel('Тип топлива')
        plt.ylabel('Цена')
        plt.grid()
        plt.show()

    def plot_mileage_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Mileage'], bins=30, kde=True)
        plt.title('Распределение пробега')
        plt.show()

    def plot_price_by_owner_count(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x='Owner_Count', y='Price', estimator='mean', marker='o')
        plt.title('Зависимость цены от количества владельцев')
        plt.xlabel('Количество владельцев')
        plt.ylabel('Средняя цена')
        plt.grid()
        plt.show()

    def plot_engine_size_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Engine_Size'], bins=30, kde=True)
        plt.title('Распределение объема двигателя')
        plt.show()

    def plot_price_vs_year_audi(self):
        # Фильтрация данных для бренда Audi
        audi_df = self.df[self.df['Brand'] == 'Audi']
        average_price_by_year = audi_df.groupby('Year')['Price'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=average_price_by_year, x='Year', y='Price')
        plt.title('Зависимость средней цены от года выпуска (Audi)')
        plt.xlabel('Год выпуска')
        plt.ylabel('Средняя цена')
        plt.grid()
        plt.show()

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, num_points=50):
        plt.figure(figsize=(10, 6))
        
        # Используем Seaborn для построения графика
        sns.lineplot(x=range(num_points), y=y_true[:num_points], label='Фактические значения', color='blue')
        sns.lineplot(x=range(num_points), y=y_pred[:num_points], label='Предсказанные значения', color='red')
        
        # Добавляем доверительные интервалы (пример)
        if model_name != "Dummy":
            plt.fill_between(range(num_points), 
                            y_pred[:num_points] - 500,  # Нижняя граница
                            y_pred[:num_points] + 500,  # Верхняя граница
                            color='red', alpha=0.2, label='Доверительный интервал')
        
        plt.title(f'Фактические и предсказанные значения по регрессору {model_name}')
        plt.xlabel('Наблюдение')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_all(self):
        self.plot_average_price_vs_mileage_audi()  
        self.plot_average_price_by_fuel_type()  
        self.plot_mileage_distribution()
        self.plot_price_by_owner_count()
        self.plot_engine_size_distribution()
        self.plot_price_vs_year_audi()                           

    def plot_combined_results(self, y_test, results, sample_size=0.1):
        sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_size), replace=False)
        y_test_sample = y_test.iloc[sample_indices]

        # Построение графика
        plt.figure(figsize=(10, 6))
        for name, result in results.items():
            y_pred_sample = result["y_pred"][sample_indices]
            sns.scatterplot(x=y_test_sample, y=y_pred_sample, alpha=0.6, label=f"{name} (R² = {result['R²']:.4f})")

        # Идеальная линия
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Идеальные предсказания')

        # Настройка графика
        plt.title("Сравнение моделей: Фактические vs Предсказанные значения", fontsize=16)
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_individual_results(self, y_test, results):
        plt.figure(figsize=(15, 10))
        for i, (name, result) in enumerate(results.items(), 1):
            plt.subplot(2, 3, i)
            sns.scatterplot(x=y_test, y=result["y_pred"], alpha=0.6)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
            plt.title(f"{name}\nR² = {result['R²']:.4f}")
            plt.xlabel("Фактические значения")
            plt.ylabel("Предсказанные значения")
            plt.grid(True)

        plt.suptitle("Сравнение моделей: Фактические vs Предсказанные значения", fontsize=16)
        plt.tight_layout()
        plt.show()    