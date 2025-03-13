from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

def get_regressor():
    return DecisionTreeRegressor(random_state=42)

def evaluate_regressor(X_train, y_train, X_test, y_test):
    regressor = get_regressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')
    return {
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
        "CV R² Mean": scores.mean(),
        "CV R² Std": scores.std(),
        "y_pred": y_pred  # Добавляем предсказанные значения
    }

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'max_depth': [None, 10, 20, 30],  # Максимальная глубина дерева
        'min_samples_split': [2, 5, 10],  # Минимальное количество образцов для разделения узла
        'min_samples_leaf': [1, 2, 4],  # Минимальное количество образцов в листе
        'max_features': ['auto', 'sqrt', 'log2']  # Количество признаков для поиска лучшего разделения
    }
    dt = get_regressor()
    grid_search = RandomizedSearchCV(estimator=dt, param_distributions=param_grid, n_iter=10, cv=3, scoring='r2', random_state=42)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_