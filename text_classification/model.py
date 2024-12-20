import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import os

# Load dữ liệu
X_data = pickle.load(open('data/X_data.pkl', 'rb'))
y_data = pickle.load(open('data/y_data.pkl', 'rb'))
X_test = pickle.load(open('data/X_test.pkl', 'rb'))
y_test = pickle.load(open('data/y_test.pkl', 'rb'))

# Tính tf idf
tdidf_vect = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(1,1), max_df=0.8)
tdidf_vect.fit(X_data)
X_data_tfidf = tdidf_vect.transform(X_data)
X_test_tfidf = tdidf_vect.transform(X_test)
# # In ra ma trận TF-IDF của dữ liệu huấn luyện
# print("\nMa trận TF-IDF cho X_data (1-gram):")
# print(X_data_tfidf.toarray())
# # In ra tên các tính năng (từ vựng)
# print("\nCác tính năng (từ vựng) trong ma trận TF-IDF:")
# print(tdidf_vect.get_feature_names_out())

# ngram level
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram.fit(X_data)
X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)
# # In ra ma trận TF-IDF cho X_data (ngram 2-3)
# print("\nMa trận TF-IDF cho X_data (ngram 2-3):")
# print(X_data_tfidf_ngram.toarray())

# ngram-char level
tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram_char.fit(X_data)
X_data_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_data)
X_test_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_test)
# # In ra ma trận TF-IDF cho X_data (ngram char 2-3)
# print("\nMa trận TF-IDF cho X_data (ngram char 2-3):")
# print(X_data_tfidf_ngram_char.toarray())
# # Lấy các tính năng từ các vector đã tính toán
# print("\nCác tính năng (tính năng ngram char 2-3) trong ma trận TF-IDF:")
# print(tfidf_vect_ngram_char.get_feature_names_out())

# Lấy chủ đề
encoder = preprocessing.LabelEncoder()
encoder.fit(y_data)

MODEL_PATH = "models"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def train_model(X_data, y_data, X_test=None, y_test=None):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    # Lưu X_train, y_train và LabelEncoder
    print("Lưu tập tin train và test")
    # Mã hóa các nhãn
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)  
    with open(os.path.join(MODEL_PATH, 'train_data.pkl'), 'wb') as f:
        pickle.dump((X_train, y_train, label_encoder), f)
    with open(os.path.join(MODEL_PATH, 'test_data.pkl'), 'wb') as fr:
        pickle.dump((X_test, y_test, label_encoder), fr)
        # Save LabelEncoder
    with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Khởi tạo và huấn luyện mô hình Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Dự đoán trên tập huấn luyện và kiểm tra
    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_test)

    print("Training accuracy: ", metrics.accuracy_score(train_predictions, y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_test))

    train_time = time.time() - start_time
    print('Thời gian hoàn thành huấn luyện Naive Bayes là', train_time, 'giây.')
    return classifier

# Huấn luyện mô hình
text_clf = train_model(X_data_tfidf, y_data, X_test=X_test_tfidf, y_test=y_test)


# Lưu mô hình Naive Bayes đã huấn luyện
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))

print("Mô hình đã được lưu thành công.")