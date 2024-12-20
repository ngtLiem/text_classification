import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Tạo dữ liệu
# corpus = ['Dữ liệu lớn giúp chúng ta phân tích và dự đoán xu hướng trong các lĩnh vực khác nhau.',
#               'Dữ liệu phân tích giúp doanh nghiệp đưa ra quyết định chiến lược chính xác hơn.']
# corpus = ['Pháo hoa sáng rực trên bầu trời đêm',
#           'Màn pháo hoa chào đón năm mới rất đẹp']
corpus = ['Học lập trình rất vui',
          'Tôi yêu thích lập trình máy tính từ lớp 6',
          'Tôi đang sử dụng máy tính hp']
# đọc và tạo kho lưu trữ
words_set = set()
for doc in  corpus:
    words = doc.split(' ')
    words_set = words_set.union(set(words))
print('Tổng số từ có trong tài liệu:',len(words_set))
# print('Danh sách từ trong tài liệu:', words_set)
# Khơi tạo vectorizer
tf_idf_model  = TfidfVectorizer()
tf_idf_vector = tf_idf_model.fit_transform(corpus)

tf_idf_array = tf_idf_vector.toarray()
words_set = tf_idf_model.get_feature_names_out()
# print(words_set)
df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
with pd.option_context('display.float_format', '{:.4f}'.format):print(df_tf_idf)

