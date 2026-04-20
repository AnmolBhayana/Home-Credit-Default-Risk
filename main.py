"""
Home Credit Default Risk — Full ML Pipeline
=============================================
Matches the methodology from the project report:
- Data loading and exploration (7 files)
- EDA and visualisations (heatmaps, bar graphs, distribution plots, pair plots)
- Data cleaning and preprocessing
- 5 classification models: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes
- Performance evaluation: Accuracy, Precision, Recall, F1 Score, ROC Curve, Confusion Matrix
- Model comparison and selection

Dataset: Home Credit Default Risk (Kaggle)
https://www.kaggle.com/competitions/home-credit-default-risk/data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_DIR = "data/"          # folder containing all CSV files
OUTPUT_DIR = "outputs/"     # folder to save all plots and results
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 16
TEST_SIZE = 0.25

# Set to None to use the full dataset (recommended)
# Set to e.g. 50000 for faster testing on lower-spec machines
SAMPLE_SIZE = None


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    print("\n" + "="*60)
    print("  STEP 1: Loading Data")
    print("="*60)

    app_train = pd.read_csv(DATA_DIR + "application_train.csv")
    if SAMPLE_SIZE:
        app_train = app_train.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  [SAMPLE MODE] Using {SAMPLE_SIZE} rows for faster processing.")
    app_test  = pd.read_csv(DATA_DIR + "application_test.csv")
    bureau    = pd.read_csv(DATA_DIR + "bureau.csv")
    bureau_bal= pd.read_csv(DATA_DIR + "bureau_balance.csv")
    cc_bal    = pd.read_csv(DATA_DIR + "credit_card_balance.csv")
    prev_app  = pd.read_csv(DATA_DIR + "previous_application.csv")
    pos_cash  = pd.read_csv(DATA_DIR + "POS_CASH_balance.csv")

    print(f"Training data shape:      {app_train.shape}")
    print(f"Test data shape:          {app_test.shape}")
    print(f"Bureau shape:             {bureau.shape}")
    print(f"Bureau balance shape:     {bureau_bal.shape}")
    print(f"Credit card balance:      {cc_bal.shape}")
    print(f"Previous application:     {prev_app.shape}")
    print(f"POS CASH balance:         {pos_cash.shape}")

    print("\nTraining data head:")
    print(app_train.head())

    print(f"\nTarget distribution:\n{app_train['TARGET'].value_counts()}")
    print(f"Default rate: {app_train['TARGET'].mean():.2%}")

    return app_train, app_test, bureau, bureau_bal, cc_bal, prev_app, pos_cash


# ─────────────────────────────────────────────
# 2. EDA AND VISUALISATIONS
# ─────────────────────────────────────────────

def run_eda(app_train, bureau, bureau_bal):
    print("\n" + "="*60)
    print("  STEP 2: EDA and Visualisations")
    print("="*60)

    # --- 2a. Bureau Balance Heatmap ---
    print("  Generating bureau balance heatmap...")
    plt.figure(figsize=(10, 8))
    bureau_bal_numeric = bureau_bal.select_dtypes(include=[np.number])
    corr = bureau_bal_numeric.corr()
    sns.heatmap(corr, annot=True, fmt=".1f", cmap='RdYlBu_r', center=0)
    plt.title("Bureau Balance Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "01_bureau_balance_heatmap.png", dpi=150)
    plt.close()

    # --- 2b. Failure to repay by age group ---
    print("  Generating failure to repay by age group...")
    app_train['YEARS_BIRTH'] = abs(app_train['DAYS_BIRTH']) / 365
    app_train['AGE_GROUP'] = pd.cut(app_train['YEARS_BIRTH'],
                                     bins=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    age_default = app_train.groupby('AGE_GROUP', observed=True)['TARGET'].mean() * 100

    plt.figure(figsize=(10, 5))
    age_default.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title("Failure to Repay by Age Group")
    plt.xlabel("Age Group (years)")
    plt.ylabel("Failure to Repay (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "02_failure_by_age_group.png", dpi=150)
    plt.close()

    # --- 2c. Age of Client distribution ---
    print("  Generating age distribution...")
    plt.figure(figsize=(8, 4))
    app_train['YEARS_BIRTH'].plot(kind='hist', bins=30, color='steelblue',
                                   edgecolor='white')
    plt.title("Age of Client")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "03_age_histogram.png", dpi=150)
    plt.close()

    # --- 2d. Distribution of Ages (KDE) ---
    plt.figure(figsize=(8, 4))
    app_train.loc[app_train['TARGET'] == 0, 'YEARS_BIRTH'].plot(
        kind='kde', label='Repaid', color='blue')
    app_train.loc[app_train['TARGET'] == 1, 'YEARS_BIRTH'].plot(
        kind='kde', label='Default', color='red')
    plt.title("Distribution of Ages")
    plt.xlabel("Age (years)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "04_age_distribution_kde.png", dpi=150)
    plt.close()

    # --- 2e. Correlation Heatmap (EXT_SOURCE + TARGET) ---
    print("  Generating EXT_SOURCE correlation heatmap...")
    cols = ['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
    corr_sub = app_train[cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap='RdYlBu_r', center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "05_ext_source_heatmap.png", dpi=150)
    plt.close()

    # --- 2f. EXT_SOURCE distribution by target ---
    print("  Generating EXT_SOURCE distribution plots...")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for i, col in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        for target_val, color, label in [(0, 'blue', 'Repaid'), (1, 'red', 'Default')]:
            subset = app_train.loc[app_train['TARGET'] == target_val, col].dropna()
            subset.plot(kind='kde', ax=axes[i], color=color, label=label)
        axes[i].set_title(f"Distribution of {col} by Target Value")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "06_ext_source_distributions.png", dpi=150)
    plt.close()

    # --- 2g. Pair plot (EXT_SOURCE + age) ---
    print("  Generating pair plot (may take a moment)...")
    pair_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'YEARS_BIRTH', 'TARGET']
    pair_data = app_train[pair_cols].dropna().sample(3000, random_state=42)
    pair_plot = sns.pairplot(pair_data, hue='TARGET', diag_kind='kde',
                              plot_kws={'alpha': 0.4},
                              vars=['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                    'EXT_SOURCE_3', 'YEARS_BIRTH'])
    pair_plot.fig.suptitle("Ext Source and Age Features Pairs Plot", y=1.01)
    plt.savefig(OUTPUT_DIR + "07_pair_plot.png", dpi=120, bbox_inches='tight')
    plt.close()

    # --- 2h. CREDIT_INCOME_PERCENT distribution ---
    print("  Generating credit income percent distributions...")
    app_train['CREDIT_INCOME_PERCENT']   = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
    app_train['ANNUITY_INCOME_PERCENT']  = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['CREDIT_TERM']             = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
    app_train['DAYS_EMPLOYED_PERCENT']   = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(),
                        ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT',
                         'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
        for target_val, color, label in [(0, 'blue', 'Repaid'), (1, 'red', 'Default')]:
            subset = app_train.loc[app_train['TARGET'] == target_val, col].dropna()
            subset.plot(kind='kde', ax=ax, color=color, label=label)
        ax.set_title(f"Distribution of {col} by Target Value")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "08_credit_income_distributions.png", dpi=150)
    plt.close()

    # --- 2i. Bureau balance null check (from report) ---
    print("  Bureau balance null values after mean imputation:")
    for c in bureau_bal.select_dtypes(include=[np.number]).columns:
        bureau_bal[c] = bureau_bal[c].fillna(bureau_bal[c].mean())
    print(bureau_bal.isnull().sum())

    print("\n  All EDA plots saved to outputs/ folder.")
    return app_train


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(app_train):
    print("\n" + "="*60)
    print("  STEP 3: Data Cleaning and Preprocessing")
    print("="*60)

    df = app_train.copy()
    y = df['TARGET']
    X = df.drop(['TARGET'], axis=1)

    # Drop non-numeric ID column and any interval/category columns added during EDA
    drop_cols = ['SK_ID_CURR', 'AGE_GROUP']
    X = X.drop([c for c in drop_cols if c in X.columns], axis=1)

    # Encode categorical columns
    print("  Encoding categorical features...")
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()

    # Impute missing values with median (matches report)
    print("  Imputing missing values with median...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    # Scale to [0, 1]
    print("  Scaling features to [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    print(f"  Final feature matrix shape: {X.shape}")
    print(f"  Target distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y.values, feature_names


# ─────────────────────────────────────────────
# 4. MODEL TRAINING AND EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict, and compute all metrics for one model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Classification Report:")
    target_names = ['can_repay', 'cannot_repay']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("-", "")
    plt.savefig(OUTPUT_DIR + f"cm_{fname}.png", dpi=150)
    plt.close()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f"roc_{fname}.png", dpi=150)
    plt.close()

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'ROC AUC': round(roc_auc, 4)
    }


def run_models(X, y):
    print("\n" + "="*60)
    print("  STEP 4: Training and Evaluating All Models")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    results = []

    # Model 1: Logistic Regression
    lr = LogisticRegression(random_state=15, max_iter=1000)
    results.append(evaluate_model("Logistic Regression", lr,
                                   X_train, X_test, y_train, y_test))

    # Model 2: Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    results.append(evaluate_model("Decision Tree", dt,
                                   X_train, X_test, y_train, y_test))

    # Model 3: Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    results.append(evaluate_model("Random Forest", rf,
                                   X_train, X_test, y_train, y_test))

    # Model 4: K-Nearest Neighbors (k=5 selected as best per report)
    knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    results.append(evaluate_model("K-Nearest Neighbors (k=5)", knn5,
                                   X_train, X_test, y_train, y_test))

    # Model 5: Naive Bayes
    nb = GaussianNB()
    results.append(evaluate_model("Naive Bayes", nb,
                                   X_train, X_test, y_train, y_test))

    return results, lr, rf  # return best models for feature importance


# ─────────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────────

def compare_models(results):
    print("\n" + "="*60)
    print("  STEP 5: Model Comparison")
    print("="*60)

    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

    # Bar chart comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — All Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "model_comparison.png", dpi=150)
    plt.close()

    print(f"\n  Best model: {df.iloc[0]['Model']} "
          f"(Accuracy: {df.iloc[0]['Accuracy']:.4f})")
    print("\n  Per the report, Logistic Regression is selected as the final model "
          "due to minimal accuracy difference, mixed data types, and lower "
          "computational cost vs Random Forest.")

    return df


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

def feature_importance(rf_model, feature_names):
    print("\n" + "="*60)
    print("  STEP 6: Feature Importance (Random Forest)")
    print("="*60)

    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20)

    plt.figure(figsize=(10, 6))
    top20.sort_values().plot(kind='barh', color='steelblue', edgecolor='white')
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "feature_importance.png", dpi=150)
    plt.close()

    print("  Top 10 features:")
    print(top20.head(10).to_string())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏦 Home Credit Default Risk — ML Pipeline")
    print("=" * 60)

    # Step 1: Load
    app_train, app_test, bureau, bureau_bal, cc_bal, prev_app, pos_cash = load_data()

    # Step 2: EDA
    app_train = run_eda(app_train, bureau, bureau_bal)

    # Step 3: Preprocess
    X, y, feature_names = preprocess(app_train)

    # Step 4: Models
    results, lr_model, rf_model = run_models(X, y)

    # Step 5: Compare
    comparison_df = compare_models(results)

    # Step 6: Feature importance
    feature_importance(rf_model, feature_names)

    # Save comparison table
    comparison_df.to_csv(OUTPUT_DIR + "model_comparison.csv", index=False)

    print("\n✅ Pipeline complete. All outputs saved to outputs/ folder.")
    print("   Plots: confusion matrices, ROC curves, EDA visualisations")
    print("   CSV:   model_comparison.csv")
