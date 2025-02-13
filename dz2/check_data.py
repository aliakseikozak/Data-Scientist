import logging


logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s") 

class CheckData:
    logger = logging.getLogger(__name__)
    logger.info('Started modul CheckData')

    def count_missing_values(self, df): # Подсчитывает пустые или пропущенные значения в каждом столбце DataFrame
        try:
            if df is not None:
                return df.isnull().sum()
            else:
                self.logger.info("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            self.logger.exception(f"Ошибка подсчета постых значений: {e}")
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
                 #   self.logger.info("Нет пропущенных значений.")
                    return None
            else:
                self.logger.info("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            self.logger.exception(f"Ошибка вывода информации о пропущенных значениях: {e}")
            return None

    def fill_missing_values(self, df, method='mean', file_path=None):  # Заполняет пропущенные значения в DataFrame
        try:        
            if df is not None:
                has_missing_values = False
                for column in df.columns:
                    if df[column].isnull().any():
                        has_missing_values = True 
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
                                logger.info(f"Метод '{method}' не поддерживается для строковых данных. Пропущенные значения не заполнены.")
                        else:
                            logger.info(f"Тип данных в столбце '{column}' не поддерживается. Пропущенные значения не заполнены.")

                # Запись заполненных данных в файл, если указан путь
                if has_missing_values and file_path is not None:
                    df.to_csv(file_path, index=False)
                    self.logger.info(f"Данные успешно записаны в файл: {file_path}")

                return df
            else:
                self.logger.info("Отсуствуют данные в DataFrame")
                return None
        except Exception as e:
            self.logger.exception(f"Ошибка заполнения пропущенных значений: {e}")
            return None