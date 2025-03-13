import pandas as pd

def predict_car_price(best_model, encoder, scaler, input_data):
    # Преобразуем входные данные в DataFrame
    input_df = pd.DataFrame([input_data])

    # Проверка категориальных признаков
    categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
    for col in categorical_cols:
        if input_data[col] not in encoder.categories_[categorical_cols.index(col)]:
            print(f"Внимание: '{input_data[col]}' не было в обучающих данных для признака '{col}'. Результат может быть неточным.")

    # Кодируем категориальные признаки
    encoded_cols = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    # Объединяем закодированные признаки с числовыми
    numerical_cols = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']
    numerical_data = input_df[numerical_cols].reset_index(drop=True)
    X = pd.concat([numerical_data, encoded_df], axis=1)

    # Масштабируем числовые признаки
    X_scaled = scaler.transform(X)

    # Предсказание стоимости
    predicted_price = best_model.predict(X_scaled)
    return predicted_price[0]
