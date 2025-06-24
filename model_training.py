import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder


# 加载数据
df = pd.read_excel('data/保密异常事件集合.xlsx')

# 合并文本特征
text_columns = df.columns[:10]
df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# 文本向量化
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['combined_text'])
y = df[df.columns[10]]
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 转换为0/1数值标签

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 定义评估指标
def evaluate_model(model):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            if len(np.unique(y_test)) > 2 
            else roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    }

# 模型对比
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': XGBClassifier(
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1
    )

}


# 超参数网格
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'SVM': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
}

best_model = None
best_score = 0

# 训练和调优
for name in models:
    grid_search = GridSearchCV(models[name], param_grids[name], cv=3, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    print(f"{name} 最佳参数: {grid_search.best_params_}")
    scores = evaluate_model(grid_search.best_estimator_)
    print(f"{name} 评估结果:\n{pd.Series(scores).to_string()}\n")
    
    if scores['F1'] > best_score:
        best_score = scores['F1']
        best_model = grid_search.best_estimator_

# 保存最佳模型
print(f"最优模型: {type(best_model).__name__}\n综合表现:\n{pd.Series(evaluate_model(best_model)).to_string()}")
XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 结果输出
# Logistic Regression 最佳参数: {'C': 10}
# Logistic Regression 评估结果:
# Accuracy     1.0
# Precision    1.0
# Recall       1.0
# F1           1.0
# AUC          1.0

# Random Forest 最佳参数: {'max_depth': None, 'n_estimators': 100}
# Random Forest 评估结果:
# Accuracy     1.0
# Precision    1.0
# Recall       1.0
# F1           1.0
# AUC          1.0

# SVM 最佳参数: {'C': 1, 'kernel': 'linear'}
# SVM 评估结果:
# Accuracy     1.0
# Precision    1.0
# Recall       1.0
# F1           1.0
# AUC          1.0

# XGBoost 最佳参数: {'learning_rate': 0.1, 'max_depth': 3}
# XGBoost 评估结果:
# Accuracy     0.987342
# Precision    0.987506
# Recall       0.987342
# F1           0.986117
# AUC          1.000000

# 最优模型: LogisticRegression
# 综合表现:
# Accuracy     1.0
# Precision    1.0
# Recall       1.0
# F1           1.0
# AUC          1.0
