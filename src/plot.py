import matplotlib.pyplot as plt

def plot_dataset(df):
    """
    Plot a scatter plot of Hours vs Scores.
    """
    plt.figure()
    plt.scatter(df['Hours'], df['Scores'])
    plt.title('Hours vs Scores')
    plt.xlabel('Hours Studied')
    plt.ylabel('Scores')
    plt.show(block=False)

def plot_eval(X, y, model):
    """
    Plot the actual data and the regression line.
    """
    plt.figure()
    plt.scatter(X, y, color='blue', label='Actual Scores')
    plt.plot(X, model.predict(X), color='red', label='Predicted Score')
    plt.title('Linear Regression')
    plt.xlabel('Hours Studied')
    plt.ylabel('Scores')
    plt.legend()
    plt.show()
