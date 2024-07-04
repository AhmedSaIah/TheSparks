import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))

def plot_decision_tree(model, X):
    try:
        tree_to_plot = model.estimators_[0]
        dot_data = export_graphviz(
            tree_to_plot,
            out_file=None,
            feature_names=X.columns,
            class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        return graph
    except FileNotFoundError as e:
        print("Graphviz executable not found. Please ensure Graphviz is installed and added to the system PATH.")
        return None

def view_decision_tree(graph):
    if graph:
        graph.view()
    else:
        print("Unable to render the decision tree due to missing Graphviz.")