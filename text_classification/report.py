import pandas as pd
import collections
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


MODEL_PATH = "models"
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load X_train, y_train và LabelEncoder
with open(os.path.join(MODEL_PATH, 'train_data.pkl'), 'rb') as f:
    X_train, y_train, label_encoder = pickle.load(f)

with open(os.path.join(MODEL_PATH, 'test_data.pkl'), 'rb') as f:
    X_test, y_test, label_encoder = pickle.load(f)


# Tạo báo cáo dữ liệu mẫu tập tin huấn luyện train và test
import pandas as pd
import collections

# Đếm số lượng mỗi nhãn trong tập train và test
label_counts_train = collections.Counter(y_train)
label_counts_test = collections.Counter(y_test)

# Tổng train và test
total_labels = list(y_train) + list(y_test)

# Đếm số lượng mỗi nhãn trong tổng số dữ liệu
total_label_counts = collections.Counter(total_labels)

# Chuẩn bị dữ liệu cho DataFrame
data = []
for label_idx, total_count in total_label_counts.items():
    label = label_encoder.inverse_transform([label_idx])
    train_count = label_counts_train.get(label_idx, 0)
    test_count = label_counts_test.get(label_idx, 0)
    train_ratio = 100 * (train_count / total_count if total_count > 0 else 0)
    test_ratio = 100 * (test_count / total_count if total_count > 0 else 0) 
    data.append([label[0], total_count, train_count, train_ratio, test_count, test_ratio])

# Tạo DataFrame
df = pd.DataFrame(data, columns=['Label', 'Total', 'Train', '% Train', 'Test', '% Test'])

# Hiển thị DataFrame
print(df)


# show kết quả trên từng nhãn
MODEL_PATH = "models"
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

from sklearn.metrics import classification_report
nb_model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
y_pred = nb_model.predict(X_test)

print('Thuật toán Naive Bayes có độ chính xác =', np.mean(y_pred == y_test))

report = classification_report(y_test, y_pred, target_names=list(label_encoder.classes_), output_dict=True)
# Chuyển  sang DataFrame để định dạng
report_df = pd.DataFrame(report).transpose()
print("\nKết quả dữ liệu huấn luyện của từng nhãn:")
print(report_df.to_string())
