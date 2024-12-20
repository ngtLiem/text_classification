import os
from collections import Counter
import pickle

# hàm đếm số lần xuất hiện của các từ
def count_word_frequencies(lines):
    word_counts = Counter()
    for line in lines:
        words = line.strip().split()
        word_counts.update(words)
    return word_counts

# Đọc dữ liệu từ tệp
def read_file(file_path, encoding='utf-16'):
    with open(file_path, 'r', encoding=encoding) as f:
        return f.readlines()

# Ghi dữ liệu vào tệp
def write_file(file_path, lines, encoding='utf-16'):
    with open(file_path, 'w', encoding=encoding) as f:
        f.writelines(lines)

# Lấy dữ liệu
x_data_lines = read_file('data/data_txt/X_data.txt')
X_test_lines = read_file('data/data_txt/X_test.txt')

# Đếm tần suất các từ trong dữ liệu
word_frequencies = count_word_frequencies(x_data_lines)
sorted_word_freqs = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

# Tạo danh sách từ stopword từ 100 từ xuất hiện nhiều nhất
stopwords = set()
# Lưu các từ và tần suất vô stopwords.txt
with open('data/stopwords.txt', 'w', encoding='utf-16') as f:
    for word, freq in sorted_word_freqs[:100]:
        stopwords.add(word)
        f.write(f"{word}: {freq}\n")

# Danh sách từ quan trọng
important_words = set(['người', 'việc', 'nhà', 'tiền', 'điều', 'năm', 'tháng', 'ngày', 'số', 'việt_nam', 'đội', 'trận', 'anh', 'chị', 'tôi'])

def remove_stopwords(line):
    return ' '.join(word for word in line.strip().split() if word not in stopwords or word in important_words)

# Xử lý và ghi dữ liệu đã loại bỏ stopwords
x_data_processed = [remove_stopwords(line) + '\n' for line in x_data_lines]
X_test_processed = [remove_stopwords(line) + '\n' for line in X_test_lines]

write_file('data/X_data.txt', x_data_processed)
write_file('data/X_test.txt', X_test_processed)

# Lưu các dòng đã xử lý vào file pkl
def save_to_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Lưu dữ liệu X_data, y_data, X_test, y_test vào file pkl
save_to_pickle('data/X_data.pkl', x_data_processed)
save_to_pickle('data/y_data.pkl', read_file('data/data_txt/y_data.txt'))
save_to_pickle('data/X_test.pkl', X_test_processed)
save_to_pickle('data/y_test.pkl', read_file('data/data_txt/y_test.txt'))

final_stopwords = stopwords - important_words
with open('data/stopwords_without_important.txt', 'w', encoding='utf-16') as f:
    for word in final_stopwords:
        f.write(f"{word}\n")

     
# # Giao dien xoa tu dung
# import tkinter as tk
# from tkinter import messagebox
# from underthesea import word_tokenize

# def process_text():
#     input_text = text_input.get("1.0", tk.END).strip()
#     if not input_text:
#         messagebox.showwarning("Lỗi: ", "Vui lòng nhập văn bản để tách từ.")
#         return
#     tokens = word_tokenize(input_text, format="text").lower()
#     text = remove_stopwords(tokens)
#     result_output.delete("1.0", tk.END)
#     result_output.insert(tk.END, text)

# # Tạo  giao diện chính
# root = tk.Tk()
# root.title("Xóa các từ dừng")


# # Label nhập văn bản
# input_label = tk.Label(root, text="Nhập văn bản cần loại bỏ từ dùng:")
# input_label.pack(pady=5)

# # Textbox nhập văn bản
# text_input = tk.Text(root, height=8, width=60, font=("Arial", 14))
# text_input.pack(pady=5)

# # Nút xử lý văn bản
# process_button = tk.Button(root, text="Xóa từ dừng", command=process_text)
# process_button.pack(pady=10)

# # Label hiển thị kết quả
# result_label = tk.Label(root, text="Kết quả văn bản sau khi xóa từ dừng:")
# result_label.pack(pady=5)

# # Textbox hiển thị kết quả
# result_output = tk.Text(root, height=8, width=60, state=tk.NORMAL, font=("Arial", 14))
# result_output.pack(pady=5)

# # Nút thoát chương trình
# exit_button = tk.Button(root, text="Thoát", command=root.destroy)
# exit_button.pack(pady=10)

# root.mainloop()