import tkinter as tk
from tkinter import messagebox
from underthesea import word_tokenize
from stopword import remove_stopwords
from text_preprocessing import text_preprocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Hàm tách từ
def tokenize_text():
    input_text = text_input.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Lỗi", "Vui lòng nhập văn bản vào ô trên.")
        return

    # Tách từ
    tokens = word_tokenize(input_text, format="text").lower()
    result_text.delete("1.0", tk.END)
    result_text.insert("1.0", tokens)

# Hàm xóa từ dừng
def rv_stopwords():
    input_text = text_input.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Lỗi", "Vui lòng nhập văn bản vào ô trên.")
        return
    tokens = word_tokenize(input_text, format="text").lower()
    filtered_tokens = remove_stopwords(tokens)

    result_text.delete("1.0", tk.END)
    result_text.insert("1.0", filtered_tokens)

# Hàm tiền xử lý dữ liệu
def data_preprocessing():
    input_data = text_input.get("1.0", tk.END).strip()
    if not input_data:
        messagebox.showerror("Lỗi", "Vui lòng nhập văn bản vào ô trên.")
        return
    tokens = word_tokenize(input_data, format="text").lower()
    text = remove_stopwords(tokens)
    data = text_preprocess(text)

    result_text.delete("1.0", tk.END)
    result_text.insert("1.0", data)

# Hàm tính TF-IDF
def calculate_tfidf():
    input_text = text_input.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showerror("Lỗi", "Vui lòng nhập văn bản vào ô trên.")
        return

    # Chuyển văn bản thành danh sách các câu
    corpus = [sentence.strip() for sentence in input_text.split(".") if sentence.strip()]

    if not corpus:
        messagebox.showerror("Lỗi", "Văn bản không có nội dung hợp lệ để tính TF-IDF.")
        return

    # Tính TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_array = tfidf_matrix.toarray()
    words = vectorizer.get_feature_names_out()

    # Tạo DataFrame để hiển thị
    df_tfidf = pd.DataFrame(tfidf_array, columns=words)

    # Hiển thị kết quả
    result_text.delete("1.0", tk.END)
    result_text.insert("1.0", df_tfidf.to_string())

# Tạo giao diện
root = tk.Tk()
root.title("Tiền xử lý dữ liệu")
root.geometry("600x500")

# khung nhập liệu
frame_input = tk.Frame(root)
frame_input.pack(pady=10, fill="x")

label_input = tk.Label(frame_input, text="Nhập vào văn bản:", font=("Arial", 12))
label_input.pack(anchor="w")

text_input = tk.Text(frame_input, height=10, wrap="word", font=("Arial", 12))
text_input.pack(fill="x")

# nút thực hiện
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

button_tokenize = tk.Button(frame_buttons, text="Tách từ", font=("Arial", 12), command=tokenize_text)
button_tokenize.pack(side="left", padx=10)

button_remove_stopwords = tk.Button(frame_buttons, text="Xóa từ dừng", font=("Arial", 12), command=rv_stopwords)
button_remove_stopwords.pack(side="left", padx=10)

button_data_preprocessing = tk.Button(frame_buttons, text="Tiền xử lý dữ liệu", font=("Arial", 12), command=data_preprocessing)
button_data_preprocessing.pack(side="left", padx=10)

button_tfidf = tk.Button(frame_buttons, text="Tính TF-IDF", font=("Arial", 12), command=calculate_tfidf)
button_tfidf.pack(side="left", padx=10)

# hiển thị kết quả
frame_result = tk.Frame(root)
frame_result.pack(fill="both", expand=True, pady=10)

label_result = tk.Label(frame_result, text="Kết quả:", font=("Arial", 12))
label_result.pack(anchor="w")

result_text = tk.Text(frame_result, height=10, wrap="word", font=("Arial", 12))
result_text.pack(fill="x", pady=5)

# Chạy ứng dụng
root.mainloop()