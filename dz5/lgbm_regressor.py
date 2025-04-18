from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

def get_regressor():
    return LGBMRegressor(random_state=42, verbosity=-1)

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
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }
    lgbm = get_regressor()
    grid_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_grid, n_iter=10, cv=3, scoring='r2', random_state=42)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_