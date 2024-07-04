from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train):
    """
    Fit the Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def eval_model(model, X_test, y_test):
    """
    Evaluate the Linear Regression model with Mean Squared Error and R² Score.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
