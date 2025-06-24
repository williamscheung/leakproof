import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_and_save_model():
    
    # 加载数据
    df = pd.read_excel('data/保密异常事件集合.xlsx')
    y = df.iloc[:, 10]  # 第11列为标签列
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 合并文本特征
    text_columns = df.columns[:10]
    X = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # 转换为0/1数值标签

    # df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # 文本向量化
    # tfidf = TfidfVectorizer(max_features=5000)
    # X = tfidf.fit_transform(df['combined_text'])
        # le = LabelEncoder()
    # y_encoded = le.fit_transform(y)  # 转换为0/1数值标签
    
    
    # 创建处理流水线（TF-IDF向量化 + 分类模型）
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000,C=10))
    ])
    
    # 训练模型
    model.fit(X,y_encoded)


    # 持久化模型
    joblib.dump(model, 'leakproof_model.pkl')
    print("Model trained and saved successfully")

if __name__ == '__main__':
    train_and_save_model()
