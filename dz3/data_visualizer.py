import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s")

class DataVisualizer:
    def __init__(self, df):
        self.df = df
    logger = logging.getLogger(__name__)
    logger.info('Started modul DataVisualizer')        

    def add_histogram(self, column, bins=100, alpha=1): #Добавляет гистограмму для заданного столбца
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(self.df[column].dropna(), bins=bins, alpha=alpha, color='blue', edgecolor='black')
            plt.title(f'Гистограмма: Количество машин по цене')
            plt.xlabel('Цена')
            plt.ylabel('Количество автомобилей')
            plt.grid(axis='y', alpha=alpha)
            plt.show()
            self.logger.info("График гистограммы успешно построен")
        except Exception as e:
            self.logger.exception(f"Ошибка построения гистрограммы: {e}")
            return None

    def add_line_plot(self, x_column, y_column, color_column=None):  # Добавляет линейный график для заданных столбцов
        try:
            plt.figure(figsize=(10, 6))
            
            if color_column is not None:
                plt.scatter(self.df[x_column], self.df[y_column], c=self.df[color_column], cmap='viridis', marker='o')
            else:
                plt.plot(self.df[x_column], self.df[y_column], marker='o')
            
            plt.title(f'Линейный график: Стоимость автомобилей по годам (в цветовой гамме)')
            plt.xlabel('Количество выпущенных автомобилей')
            plt.ylabel('Год выпуска')
            plt.grid()
            plt.colorbar(label='Цветовая гамма цены автомобилей') if color_column else None  # Добавляем цветовую шкалу, если есть color_column
            plt.show()
            self.logger.info("Линейный график успешно построен")
        except Exception as e:
            self.logger.exception(f"Ошибка построения линейного графика: {e}")
            return None

    def add_scatter_plot(self, x_column, y_column): #Добавляет диаграмму рассеяния для заданных столбцов
        try:
            plt.figure(figsize=(10, 6))
            grouped_data = self.df.groupby(x_column)[y_column].mean().reset_index()
            plt.scatter(grouped_data[x_column], grouped_data[y_column], alpha=0.7)   
            #plt.scatter(self.df[x_column], self.df[y_column], alpha=0.7)        
            plt.title(f'Диаграмма рассеяния: Средняя цена автомобилей по годам')
            plt.xlabel('Год')
            plt.ylabel('Цена')
            plt.grid()
            plt.show()
            self.logger.info("Диаграмма рассеяния успешно построен")
        except Exception as e:
            self.logger.exception(f"Ошибка построения диаграммы рассеяния: {e}")
            return None