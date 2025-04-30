# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\K.Rajendra Prasad\Desktop\ML project\archive\train.csv')

# Use a subset of data for faster processing (you can scale up later)
print("Sampling dataset for faster processing...")
df = df.sample(20000, random_state=42)  # Use 20,000 samples for quick testing

# Define features and target
X = df['comment_text']  # Input feature
y = df['toxic']         # Target variable

# Convert text data to numerical form using TfidfVectorizer
print("Transforming text data with TfidfVectorizer...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')  # Use top 5000 features
X_tfidf = tfidf.fit_transform(X)  # Keep as sparse matrix

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.33, random_state=42)

# Define hyperparameters to tune
hyperparameters = [
    {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2},
    {'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 4}
]

# Train and evaluate the decision tree with each hyperparameter set
print("Training and evaluating models...")
best_accuracy = 0
best_params = None
best_tree = None

for params in hyperparameters:
    tree = DecisionTreeClassifier(**params, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Parameters: {params}, Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        best_tree = tree

# Print the best hyperparameters and accuracy
print(f"\nBest Parameters: {best_params}, Best Accuracy: {best_accuracy:.4f}")

# Visualize the best decision tree
print("Visualizing the best decision tree...")
plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    filled=True,
    feature_names=tfidf.get_feature_names_out(),
    class_names=['Not Toxic', 'Toxic'],
    rounded=True
)
plt.title('Best Decision Tree')
plt.show()

# Evaluate the best tree on the test set
print("Evaluating the best decision tree...")
y_pred_best = best_tree.predict(X_test)
print("Best Decision Tree - Classification Report")
print(classification_report(y_test, y_pred_best, target_names=['Not Toxic', 'Toxic']))
print("Best Decision Tree - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("Best Decision Tree - Accuracy Score:")
print(accuracy_score(y_test, y_pred_best))
