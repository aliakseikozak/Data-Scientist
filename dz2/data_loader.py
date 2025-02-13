import pandas as pd
import requests
import os
import matplotlib.pyplot as plt

class DataLoader:
    def load_csv(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            return None

    def load_json(self, file_path):
        try:
            data = pd.read_json(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при загрузке JSON: {e}")
            return None

    def load_api(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return pd.json_normalize(data)
        except Exception as e:
            print(f"Ошибка при загрузке из API: {e}")
            return None

    def count_missing_values(self, df): # Подсчитывает пустые или пропущенные значения в каждом столбце DataFrame
        try:
            if df is not None:
                return df.isnull().sum()
            else:
                print("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            print(f"Ошибка подсчета постых значений: {e}")
            return None


    def report_missing_values(self, df): #Выводит отчет с информацией о пропущенных значениях
        try:
            if df is not None:
                missing_counts = self.count_missing_values(df)
                total = df.shape[0]
                report = missing_counts[missing_counts > 0].reset_index()
            
                # Проверяем, есть ли пропущенные значения
                if not report.empty:
                    report.columns = ['Column', 'Missing Values']
                    report['Percentage'] = (report['Missing Values'] / total) * 100
                    return report
                else:
                    print("Нет пропущенных значений.")
                    return None
            else:
                print("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            print(f"Ошибка вывода информации о пропущенных значениях: {e}")
            return None

    def fill_missing_values(self, df, method='mean', file_path=None):  # Заполняет пропущенные значения в DataFrame
        try:        
            if df is not None:
                for column in df.columns:
                    if df[column].isnull().any():
                        if df[column].dtype in ['int64', 'float64']:  # Числовые данные
                            if method == 'mean':
                                df[column] = df[column].fillna(df[column].mean())  # Среднее значение
                            elif method == 'median':
                                df[column] = df[column].fillna(df[column].median())  # Медиана 
                            elif method == 'mode':
                                df[column] = df[column].fillna(df[column].mode()[0])  # Мода
                        elif df[column].dtype == 'object':  # Строковые данные
                            if method == 'mode':
                                df[column] = df[column].fillna(df[column].mode()[0])
                            else:
                                print(f"Метод '{method}' не поддерживается для строковых данных. Пропущенные значения не заполнены.")
                        else:
                            print(f"Тип данных в столбце '{column}' не поддерживается. Пропущенные значения не заполнены.")

                # Запись заполненных данных в файл, если указан путь
                if file_path is not None:
                    df.to_csv(file_path, index=False)
                    print("Данные успешно записаны в файл:", file_path)

                return df
            else:
                print("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            print(f"Ошибка заполнения пропущенных значений: {e}")
            return None
        
class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def add_histogram(self, column, bins=100, alpha=1): #Добавляет гистограмму для заданного столбца
        try:
            plt.figure()
            plt.hist(self.df[column].dropna(), bins=bins, alpha=alpha)
            plt.title(f'Гистограмма: {column}')
            plt.xlabel(column)
            plt.ylabel('Частота')
            plt.grid(axis='y', alpha=alpha)
            plt.show()
        except Exception as e:
            print(f"Ошибка построения гистрограммы: {e}")
            return None

    def add_line_plot(self, x_column, y_column, color_column=None):  # Добавляет линейный график для заданных столбцов
        try:
            plt.figure()
            
            if color_column is not None:
                plt.scatter(self.df[x_column], self.df[y_column], c=self.df[color_column], cmap='viridis', marker='o')
            else:
                plt.plot(self.df[x_column], self.df[y_column], marker='o')
            
            plt.title(f'Линейный график: {y_column} по {x_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid()
            plt.colorbar(label=color_column) if color_column else None  # Добавляем цветовую шкалу, если есть color_column
            plt.show()
        except Exception as e:
            print(f"Ошибка построения линейного графика: {e}")
            return None

    def add_scatter_plot(self, x_column, y_column): #Добавляет диаграмму рассеяния для заданных столбцов
        try:
            plt.figure()
            plt.scatter(self.df[x_column], self.df[y_column], alpha=0.7)        
            plt.title(f'Диаграмма рассеяния: {y_column} по {x_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid()
            plt.show()
        except Exception as e:
            print(f"Ошибка построения диаграммы рассеяния: {e}")
            return None
