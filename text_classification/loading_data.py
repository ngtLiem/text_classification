from tqdm import tqdm
from text_preprocessing import text_preprocess
import os 

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'data')

print("Quá trình đọc dữ liệu")
def get_data(folder_path):
    X = []  # dữ liệu
    y = []  # chủ đề
    dirs = os.listdir(folder_path)
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                document = f.readlines()
                document = ' '.join(document)
                document = text_preprocess(document)
                # print(document)
                X.append(document)
                y.append(path)
        #     break
        # break
    return X, y

print("Đang tải dữ liệu cho các tập dữ liệu training.")
train_path = os.path.join(dir_path, 'D:/CT551_LVTN/text_classification/data/Train_Full')
X_data, y_data = get_data(train_path)

print("Đang tải dữ liệu cho các tập dữ liệu testing")
test_path = os.path.join(dir_path, 'D:/CT551_LVTN/text_classification/data/Test_Full')
X_test, y_test = get_data(test_path)

# Lưu X_data và y_data vào file txt
print("Đang lưu dữ liệu file cho X_data.")
with open('data/data_txt/X_data.txt', 'w', encoding='utf-16') as f:
    for item in X_data:
        f.write(f"{item}\n")
print("Đang lưu dữ liệu file cho y_data.")
with open('data/data_txt/y_data.txt', 'w', encoding='utf-16') as f:
    for item in y_data:
        f.write(f"{item}\n")

# Lưu X_test và y_test
print("Đang lưu dữ liệu file cho X_test.")
with open('data/data_txt/X_test.txt', 'w', encoding='utf-16') as f:
    for item in X_test:
        f.write(f"{item}\n")
print("Đang lưu dữ liệu file cho y_test.")
with open('data/data_txt/y_test.txt', 'w', encoding='utf-16') as f:
    for item in y_test:
        f.write(f"{item}\n")