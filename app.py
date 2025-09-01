# Set Matplotlib backend to Agg (non-interactive, thread-safe)
import matplotlib
matplotlib.use('Agg')  # Must be set before importing matplotlib.pyplot or seaborn

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from flask import Flask, render_template
import os

app = Flask(__name__)

# Ensure static folder exists for saving plots
STATIC_DIR = os.path.join(app.root_path, 'static', 'plots')
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def generate_plots():
    # Clear previous plots
    for file in os.listdir(STATIC_DIR):
        os.remove(os.path.join(STATIC_DIR, file))

    # Load dataset
    data = pd.read_csv(r"C:\Users\Bharathan\Desktop\project_folder\Bank_Transaction_Fraud_Detection.csv")

    # Sample 20% of data
    data = data.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Plot 1: Class Distribution
    plt.figure(figsize=(6, 3))
    sns.countplot(x=data['Is_Fraud'], palette="coolwarm")
    plt.title("Original Class Distribution")
    plt.xlabel("Class (0: Legit | 1: Fraud)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(STATIC_DIR, 'class_distribution.png'))
    plt.close()

    # Build Transaction Graph
    G = nx.Graph()
    for _, row in data.iterrows():
        txn_id = row['Transaction_ID']
        cust_id = f"Cust_{row['Customer_ID']}"
        merch_id = f"Merch_{row['Merchant_ID']}"
        G.add_node(cust_id, type="Customer")
        G.add_node(merch_id, type="Merchant")
        G.add_node(txn_id, type="Transaction", fraud=row['Is_Fraud'], amount=row['Transaction_Amount'])
        G.add_edge(cust_id, txn_id, relationship="initiated")
        G.add_edge(txn_id, merch_id, relationship="processed_by")

    # Extract Graph Features
    pagerank_scores = nx.pagerank(G)
    clustering_scores = nx.clustering(G)
    features = []
    labels = []
    for _, row in data.iterrows():
        txn_id = row['Transaction_ID']
        amount = row['Transaction_Amount']
        pagerank = pagerank_scores.get(txn_id, 0)
        degree = G.degree(txn_id)
        clustering = clustering_scores.get(txn_id, 0)
        features.append([pagerank, degree, clustering, amount])
        labels.append(row['Is_Fraud'])

    df_features = pd.DataFrame(features, columns=["PageRank", "Degree", "Clustering", "Amount"])
    df_labels = pd.Series(labels, name="Fraud")

    # Plot 2: Feature Distributions
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_features["PageRank"], kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title("PageRank Distribution")
    sns.histplot(df_features["Degree"], kde=True, ax=axs[1], color='salmon')
    axs[1].set_title("Degree Distribution")
    sns.histplot(df_features["Clustering"], kde=True, ax=axs[2], color='lightgreen')
    axs[2].set_title("Clustering Coefficient Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_distributions.png'))
    plt.close()

    # Train Random Forest Classifier
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    # Plot 3: Confusion Matrix (Initial)
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix (Initial)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix_initial.png'))
    plt.close()

    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    clf_best = grid_search.best_estimator_
    y_pred = clf_best.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred)
    tuned_clf_report = classification_report(y_test, y_pred, output_dict=True)

    # Plot 4: Feature Importance
    importances = clf_best.feature_importances_
    feature_names = df_features.columns
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=feature_names, palette="crest")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_importance.png'))
    plt.close()

    # Plot 5: Confusion Matrix (Tuned)
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix (Tuned)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix_tuned.png'))
    plt.close()

    # Plot 6: ROC Curve
    y_probs = clf_best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(STATIC_DIR, 'roc_curve.png'))
    plt.close()

    # Plot 7: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}", color='green')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(STATIC_DIR, 'precision_recall_curve.png'))
    plt.close()

    return {
        'accuracy': accuracy,
        'tuned_accuracy': tuned_accuracy,
        'clf_report': clf_report,
        'tuned_clf_report': tuned_clf_report,
        'plot_files': [f'plots/{f}' for f in os.listdir(STATIC_DIR)]
    }

@app.route('/')
def index():
    results = generate_plots()
    return render_template('index.html', **results)

if __name__ == '__main__':
    app.run(debug=True)