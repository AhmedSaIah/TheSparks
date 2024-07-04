from src.data_loader import load_data
from src.preprocess import encode_species, split_features_labels, label_encoding
from src.model import train_test_split_data, train_model, evaluate_model
from src.plot import plot_confusion_matrix, print_classification_report, plot_decision_tree, view_decision_tree

# Load data
df = load_data()

# Preprocess data
df_encoded = encode_species(df)
X, y = split_features_labels(df_encoded)
y_labels = label_encoding(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split_data(X, y_labels)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
y_pred, accuracy = evaluate_model(model, X_test, y_test)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)

# Visualize decision tree
graph = plot_decision_tree(model, X)
view_decision_tree(graph)
