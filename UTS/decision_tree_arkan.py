import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_data(path: str, label_col: str = None):
    """
    Membaca dataset dan memisahkan fitur dengan label
    
    Parameters:
    - path (str): Path ke file CSV
    - label_col (str): Nama kolom label (jika None akan mencoba menebak)
    
    Returns:
    - X (DataFrame): Fitur-fitur
    - y (array): Label yang sudah diencode
    - class_names (list): Nama kelas asli
    """
    df = pd.read_csv(path)
    
    # Jika label_col tidak ditentukan, coba tebak kolom label
    if label_col is None:
        possible_labels = ['label', 'name', 'class', 'target']
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            raise ValueError("Tidak dapat menentukan kolom label. Harap tentukan secara manual.")
    
    print(f"Menggunakan kolom '{label_col}' sebagai label")
    print("Nilai unik label:", df[label_col].unique())
    
    X = df.drop(columns=[label_col])
    le = LabelEncoder().fit(df[label_col])
    y = le.transform(df[label_col])
    return X, y, le.classes_.tolist()


def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    """
    Membuat dan menyimpan plot feature importance
    
    Parameters:
    - model: Model decision tree yang sudah di-fit
    - feature_names: Daftar nama fitur
    - filename: Nama file output
    """
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importance[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main(data_path: str):
    # 1. Load & split data
    try:
        X, y, class_names = load_data(data_path, label_col="name")

        class_translation = {'orange': 'jeruk', 'grapefruit': 'anggur'}
        class_names = [class_translation.get(name, name) for name in class_names]
        
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # 3. Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 4. Save outputs
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    plot_feature_importance(model, X.columns.tolist())  # Perhatikan parameter yang benar
    joblib.dump(model, "model_dt.joblib")
    print("\nModel saved as model_dt.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Classifier for Citrus Fruits")
    parser.add_argument("--data", required=True, help="Path to CSV file (citrus.csv)")
    args = parser.parse_args()
    main(args.data)