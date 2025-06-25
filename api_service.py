from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# 加载预训练模型
model = joblib.load('leakproof_model.pkl')

@app.route('/', methods=['GET'])
def index():
    """显示文件上传页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 接收上传文件
    if 'file' not in request.files:
        return render_template('error.html', message='未选择文件')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message='无效的文件')
    
    # 保存并处理文件
    temp_path = 'temp_upload.xlsx'
    result_path = f'prediction_result_{datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx'
    file.save(temp_path)
    
    try:
        # 执行预测
        df = pd.read_excel(temp_path)
        text_columns = df.columns[:10]
        X = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        predictions = model.predict(X)
        df['是否真实事件'] = np.where(predictions == 1, '是', '否')
        
        # 确保下载目录存在
        os.makedirs('download', exist_ok=True)
        
        # 保存到下载目录
        full_path = os.path.join('download', result_path)
        df.to_excel(full_path, index=False)

    except Exception as e:
        return render_template('error.html', message=f'文件处理错误: {str(e)}')
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return render_template('result.html', result_file=result_path)

# 修改为
@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join('download', filename), as_attachment=True)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

