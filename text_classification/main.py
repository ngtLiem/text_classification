import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd

# Đường dẫn thư mục lưu trữ mô hình
MODEL_PATH = "models"
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Tải dữ liệu test
try:
    with open(os.path.join(MODEL_PATH, 'test_data.pkl'), 'rb') as f:
        X_test, y_test, label_encoder = pickle.load(f)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file test_data.pkl. Vui lòng kiểm tra lại.")
    exit()

# Tải mô hình Naive Bayes
try:
    model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
    if model is None:
        raise ValueError("Mô hình không tồn tại hoặc bị lỗi.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file naive_bayes.pkl. Vui lòng huấn luyện lại mô hình.")
    exit()

# độ chính xác
try:
    y_pred = model.predict(X_test)
    print('Naive Bayes, Độ chính xác =', np.mean(y_pred == y_test))
    report = classification_report(y_test, y_pred, target_names=list(label_encoder.classes_), output_dict=True)
    # Chuyển báo cáo sang DataFrame để định dạng
    report_df = pd.DataFrame(report).transpose()
    print("\nBáo cáo chi tiết:\n")
    print(report_df.to_string())
except Exception as e:
    print("Lỗi trong quá trình dự đoán:", e)
    exit()

# Tải và xử lý dữ liệu TF-IDF
try:
    X_data = pickle.load(open('data/X_data.pkl', 'rb'))
    tdidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
    tdidf_vect.fit(X_data)

except FileNotFoundError:
    print("Lỗi: Không tìm thấy file X_data.pkl. Vui lòng kiểm tra lại dữ liệu.")
    exit()

import tkinter as tk
from tkinter import messagebox

# Hàm phân loại văn bản từ giao diện người dùng
def classify_text():
    from text_preprocessing import text_preprocess
    from stopword import remove_stopwords

    user_input = text_entry.get("1.0", tk.END).strip()
    if user_input:
        try:
            # Tiền xử lý văn bản
            processed_text = user_input.lower()
            processed_text = text_preprocess(user_input)
            processed_text = remove_stopwords(processed_text)

            test_doc_tfidf = tdidf_vect.transform([processed_text])
            prediction = model.predict(test_doc_tfidf)
            prediction_label = label_encoder.inverse_transform(prediction)

            # Hiển thị nhãn của văn  bản 
            result_label.config(text=f"Chủ đề cho văn bản là: {prediction_label[0]}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi trong quá trình phân loại: {e}")
    else:
        messagebox.showwarning("Lỗi", "Vui lòng nhập văn bản để phân loại.")

# Xây dựng giao diện Tkinter
app = tk.Tk()
app.title("Phân loại văn bản với Naive Bayes")
app.geometry("500x400")

label = tk.Label(app, text="Nhập văn bản bạn cần phân loại:")
label.pack(pady=10)

text_entry = tk.Text(app, height=15, width=50)
text_entry.pack()

classify_button = tk.Button(app, text="Phân loại", command=classify_text)
classify_button.pack(pady=10)

result_label = tk.Label(app, text="", fg="red", font=("Arial", 14), wraplength=400, justify="center")
result_label.pack(pady=10)

app.mainloop()
