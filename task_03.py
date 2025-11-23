import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("===== TASK 03: Decision Tree Classifier =====")

# --- 1. Load Dataset ---
file_path = r"C:\Users\SAAD\OneDrive\Desktop\powerbi\task1folder\bank-additional\bank-additional-full.csv"

try:
    df = pd.read_csv(file_path, sep=';')
    print("File loaded successfully.")
except Exception as e:
    print("❌ ERROR: File not found or unreadable.")
    print("Details:", e)
    exit()

print("Dataset shape:", df.shape)

# --- 2. Preprocess Data ---
# Convert target column to binary
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

print("After encoding, feature count:", X_encoded.shape[1])

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# --- 4. Train Decision Tree ---
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

print("\nModel training complete.")

# --- 5. Predictions ---
y_pred = clf.predict(X_test)

# --- 6. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# --- 7. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 8. Decision Tree Plot ---
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X_encoded.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Structure")
plt.show()

print("===== DONE =====")
