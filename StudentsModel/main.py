from src.preprocess import load_data
from src.plot import plot_dataset, plot_eval
from src.model import train_model, eval_model
from sklearn.model_selection import train_test_split
import matplotlib as plt

# Load Data
df = load_data('data/dataset.csv')
print(df.head())
# Prepare Data
X = df[['Hours']]
y = df['Scores']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = train_model(X_train, y_train)

# Evaluate the Model
eval_model(model, X_test, y_test)

# Plot the Dataset
plot_dataset(df)

# Plot Evaluation Results
plot_eval(X, y, model)

plt.show()