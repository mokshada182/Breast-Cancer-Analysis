"""
Breast Cancer Analysis — Wisconsin Dataset
Ensemble Learning with Multiple Models
Author: Pragyaa Ray
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.pipeline import Pipeline

# Individual Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               VotingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  BREAST CANCER ANALYSIS — WISCONSIN DATASET")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"\n Dataset Shape  : {X.shape}")
print(f" Features       : {X.shape[1]}")
print(f" Samples        : {X.shape[0]}")
print(f" Classes        : Malignant (0) = {sum(y==0)}, Benign (1) = {sum(y==1)}")
print(f" Missing Values : {X.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 2. EDA PLOTS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#E24B4A', '#1D9E75']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — Wisconsin Breast Cancer Dataset',
             fontsize=15, fontweight='bold', y=1.01)

# Class distribution
ax = axes[0, 0]
counts = y.value_counts()
bars = ax.bar(['Malignant', 'Benign'], [counts[0], counts[1]],
               color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({count/len(y)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
ax.set_ylim(0, 420)

# Feature correlation heatmap (top 10 features)
ax = axes[0, 1]
top_features = X.corrwith(y).abs().nlargest(10).index
corr_matrix = X[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap='RdYlGn_r',
            annot=False, linewidths=0.3, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation (Top 10)', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)

# Top feature distributions by class
ax = axes[0, 2]
top_feat = X.corrwith(y).abs().nlargest(1).index[0]
for cls, color, label in zip([0, 1], colors, ['Malignant', 'Benign']):
    ax.hist(X[top_feat][y == cls], bins=25, alpha=0.65,
            color=color, label=label, edgecolor='white')
ax.set_title(f'Distribution: {top_feat[:20]}', fontweight='bold')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.legend()

# Box plots for top 4 features
ax = axes[1, 0]
top4 = X.corrwith(y).abs().nlargest(4).index.tolist()
df_melt = pd.melt(pd.concat([X[top4], y], axis=1),
                  id_vars='target', var_name='Feature', value_name='Value')
df_melt['Class'] = df_melt['target'].map({0: 'Malignant', 1: 'Benign'})
palette = {'Malignant': '#E24B4A', 'Benign': '#1D9E75'}
sns.boxplot(data=df_melt, x='Feature', y='Value', hue='Class',
            palette=palette, ax=ax, linewidth=0.8)
ax.set_title('Top 4 Features by Class', fontweight='bold')
ax.set_xticklabels([f[:12] for f in top4], rotation=20, fontsize=8)
ax.legend(fontsize=8)

# Feature importance placeholder (filled after RF trains)
ax_feat_imp = axes[1, 1]
ax_feat_imp.set_title('Random Forest Feature Importance', fontweight='bold')

# Violin plot
ax = axes[1, 2]
feat2 = X.corrwith(y).abs().nlargest(2).index[1]
df_viol = pd.DataFrame({'Value': X[feat2], 'Class': y.map({0:'Malignant',1:'Benign'})})
sns.violinplot(data=df_viol, x='Class', y='Value', palette=palette, ax=ax, inner='box')
ax.set_title(f'Violin: {feat2[:22]}', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n [Saved] 01_eda.png")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n Train/Test Split: {X_train.shape[0]} / {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 4. INDIVIDUAL BASE MODELS
# ─────────────────────────────────────────────
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF)'          : SVC(probability=True, kernel='rbf', random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes'        : GaussianNB(),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Extra Trees'        : ExtraTreesClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'AdaBoost'           : AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    'XGBoost'            : XGBClassifier(n_estimators=200, learning_rate=0.1,
                                          use_label_encoder=False, eval_metric='logloss',
                                          random_state=42, verbosity=0),
}

print("\n" + "─" * 60)
print("  BASE MODELS — CROSS-VALIDATION (5-Fold Stratified)")
print("─" * 60)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    acc   = accuracy_score(y_test, y_pred)
    roc   = roc_auc_score(y_test, y_proba)
    results[name] = {
        'model'   : model,
        'y_pred'  : y_pred,
        'y_proba' : y_proba,
        'acc'     : acc,
        'roc_auc' : roc,
        'cv_mean' : cv_scores.mean(),
        'cv_std'  : cv_scores.std(),
        'type'    : 'Base'
    }
    print(f"  {name:<22} | Acc: {acc:.4f} | ROC-AUC: {roc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# Feature importance from Random Forest
rf_model = base_models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=data.feature_names).nlargest(12)
ax_feat_imp.barh(importances.index[::-1], importances.values[::-1],
                  color='#185FA5', edgecolor='white')
ax_feat_imp.set_xlabel('Importance Score')
fig2 = plt.figure(figsize=(8, 5))
ax2 = fig2.add_subplot(111)
ax2.barh(importances.index[::-1], importances.values[::-1],
          color='#185FA5', edgecolor='white', linewidth=0.5)
ax2.set_title('Random Forest — Top 12 Feature Importances', fontweight='bold')
ax2.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/01b_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 5. ENSEMBLE MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ENSEMBLE MODELS")
print("─" * 60)

# --- Voting Classifier (Hard + Soft) ---
voting_hard = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False,
                               eval_metric='logloss', random_state=42, verbosity=0)),
        ('svm', SVC(probability=True, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ], voting='hard')

voting_soft = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False,
                               eval_metric='logloss', random_state=42, verbosity=0)),
        ('svm', SVC(probability=True, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ], voting='soft')

# --- Bagging ---
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=6),
    n_estimators=200, max_samples=0.8, max_features=0.8, random_state=42)

# --- Stacking ---
stacking = StackingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False,
                               eval_metric='logloss', random_state=42, verbosity=0)),
        ('svm', SVC(probability=True, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('et',  ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.5),
    cv=5, passthrough=False)

ensemble_models = {
    'Voting (Hard)'    : (voting_hard, 'Ensemble'),
    'Voting (Soft)'    : (voting_soft, 'Ensemble'),
    'Bagging (DT)'     : (bagging,     'Ensemble'),
    'Stacking'         : (stacking,    'Ensemble'),
}

for name, (model, mtype) in ensemble_models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    if hasattr(model, 'predict_proba') and name != 'Voting (Hard)':
        y_proba = model.predict_proba(X_test_sc)[:, 1]
    else:
        # Hard voting: use predicted labels as scores (0 or 1)
        y_proba = y_pred.astype(float)
    acc  = accuracy_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_proba)
    cv_s = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='roc_auc')
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'acc': acc, 'roc_auc': roc,
        'cv_mean': cv_s.mean(), 'cv_std': cv_s.std(), 'type': mtype
    }
    print(f"  {name:<22} | Acc: {acc:.4f} | ROC-AUC: {roc:.4f} | CV: {cv_s.mean():.4f}±{cv_s.std():.4f}")

# ─────────────────────────────────────────────
# 6. RESULTS COMPARISON PLOT
# ─────────────────────────────────────────────
df_res = pd.DataFrame([
    {'Model': k, 'Accuracy': v['acc'], 'ROC-AUC': v['roc_auc'],
     'CV Mean': v['cv_mean'], 'CV Std': v['cv_std'], 'Type': v['type']}
    for k, v in results.items()
]).sort_values('ROC-AUC', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Model Comparison — All Models', fontsize=14, fontweight='bold')

palette_type = {'Base': '#185FA5', 'Ensemble': '#1D9E75'}
bar_colors = [palette_type[t] for t in df_res['Type']]

# Accuracy
bars = axes[0].barh(df_res['Model'], df_res['Accuracy'],
                     color=bar_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, df_res['Accuracy']):
    axes[0].text(bar.get_width() - 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', ha='right', fontsize=8.5,
                 color='white', fontweight='bold')
axes[0].set_xlim(0.92, 1.01)
axes[0].set_title('Test Accuracy', fontweight='bold')
axes[0].set_xlabel('Accuracy')

# ROC-AUC
bars2 = axes[1].barh(df_res['Model'], df_res['ROC-AUC'],
                      color=bar_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars2, df_res['ROC-AUC']):
    axes[1].text(bar.get_width() - 0.003, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', ha='right', fontsize=8.5,
                 color='white', fontweight='bold')
axes[1].set_xlim(0.96, 1.005)
axes[1].set_title('ROC-AUC Score', fontweight='bold')
axes[1].set_xlabel('ROC-AUC')

legend_patches = [mpatches.Patch(color='#185FA5', label='Base Model'),
                  mpatches.Patch(color='#1D9E75', label='Ensemble Model')]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/02_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\n [Saved] 02_model_comparison.png")

# ─────────────────────────────────────────────
# 7. ROC CURVES
# ─────────────────────────────────────────────
highlight = ['Logistic Regression', 'Random Forest', 'XGBoost',
             'Gradient Boosting', 'Stacking', 'Voting (Soft)']

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.get_cmap('tab10', len(highlight))
for i, name in enumerate(highlight):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    auc = results[name]['roc_auc']
    ls = '--' if results[name]['type'] == 'Ensemble' else '-'
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
            color=cmap(i), lw=2, linestyle=ls)
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random Classifier')
ax.fill_between([0,1],[0,1],[0,1], alpha=0.04, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Key Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/03_roc_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(" [Saved] 03_roc_curves.png")

# ─────────────────────────────────────────────
# 8. CONFUSION MATRICES
# ─────────────────────────────────────────────
cms_to_plot = ['Random Forest', 'XGBoost', 'Stacking', 'Voting (Soft)']
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Confusion Matrices — Selected Models', fontsize=13, fontweight='bold')
for ax, name in zip(axes, cms_to_plot):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant','Benign'],
                yticklabels=['Malignant','Benign'],
                linewidths=0.5, cbar=False, annot_kws={'size':13,'weight':'bold'})
    acc = results[name]['acc']
    ax.set_title(f'{name}\nAcc: {acc:.4f}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/04_confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(" [Saved] 04_confusion_matrices.png")

# ─────────────────────────────────────────────
# 9. PRECISION-RECALL CURVES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for i, name in enumerate(highlight):
    prec, rec, _ = precision_recall_curve(y_test, results[name]['y_proba'])
    ap = average_precision_score(y_test, results[name]['y_proba'])
    ls = '--' if results[name]['type'] == 'Ensemble' else '-'
    ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=cmap(i), lw=2, linestyle=ls)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/05_precision_recall.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(" [Saved] 05_precision_recall.png")

# ─────────────────────────────────────────────
# 10. CROSS-VALIDATION COMPARISON
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_res))
bar_w = 0.45
b1 = ax.bar(x - bar_w/2, df_res['CV Mean'], bar_w,
            color=[palette_type[t] for t in df_res['Type']],
            yerr=df_res['CV Std'], capsize=4, edgecolor='white', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(df_res['Model'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('ROC-AUC (CV Mean ± Std)')
ax.set_title('5-Fold Cross-Validation ROC-AUC', fontsize=13, fontweight='bold')
ax.set_ylim(0.94, 1.01)
ax.legend(handles=legend_patches)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/breast_cancer_project/outputs/06_cv_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(" [Saved] 06_cv_comparison.png")

# ─────────────────────────────────────────────
# 11. BEST MODEL — DETAILED REPORT
# ─────────────────────────────────────────────
best_name = df_res.iloc[0]['Model']
best = results[best_name]
print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_name}")
print("=" * 60)
print(f"\n  Test Accuracy : {best['acc']:.4f}")
print(f"  ROC-AUC       : {best['roc_auc']:.4f}")
print(f"  CV Mean AUC   : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Malignant', 'Benign']))

# ─────────────────────────────────────────────
# 12. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FULL RESULTS SUMMARY")
print("=" * 60)
print(df_res[['Model','Type','Accuracy','ROC-AUC','CV Mean','CV Std']]
      .to_string(index=False, float_format='{:.4f}'.format))
print("\n All outputs saved to: outputs/")
print(" Done.")
