# from underthesea import word_tokenize

# while True:
#     text = input("Nhập văn bản cần tách từ (hoặc nhập 'e' để thoát): ")
#     if text.lower() == 'e':
#         break
#     tokens = word_tokenize(text, format="text").lower()
#     print("Các từ được tách ra từ câu vừa nhập vào:\n", tokens)

import tkinter as tk
from tkinter import messagebox
from underthesea import word_tokenize

def process_text():
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Lỗi: ", "Vui lòng nhập văn bản để tách từ.")
        return
    tokens = word_tokenize(input_text, format="text").lower()
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, tokens)

# Tạo  giao diện chính
root = tk.Tk()
root.title("Tách từ với Underthesea")

# Label nhập văn bản
input_label = tk.Label(root, text="Nhập văn bản cần tách từ:")
input_label.pack(pady=5)

# Textbox nhập văn bản
text_input = tk.Text(root, height=8, width=60, font=("Arial", 14))
text_input.pack(pady=5)

# Nút xử lý văn bản
process_button = tk.Button(root, text="Tách Từ", command=process_text)
process_button.pack(pady=10)

# Label hiển thị kết quả
result_label = tk.Label(root, text="Kết quả sau khi tách từ:")
result_label.pack(pady=5)

# Textbox hiển thị kết quả
result_output = tk.Text(root, height=8, width=60, font=("Arial", 14), state=tk.NORMAL)
result_output.pack(pady=5)

# Nút thoát chương trình
exit_button = tk.Button(root, text="Thoát", command=root.destroy)
exit_button.pack(pady=10)

root.mainloop()
