# -*- coding: utf-8 -*-
#수정예정

import os 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정
relative_path = "../../data/processed/Hannanum_dataset.csv"

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
file_path = os.path.join(script_dir, relative_path)

# Excel 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(file_path)

print(df[:1])

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['hannanum_nouns'].apply(lambda x: ' '.join(x)))

# 중요한 단어들을 'tfidf' 열에 추가
important_words = tfidf_vectorizer.get_feature_names_out()
word_tfidf_mapping = {}

for i, row in enumerate(tfidf_matrix):
    word_tfidf_mapping[i] = {}
    for j, word_index in enumerate(row.indices):
        word = important_words[word_index]
        tfidf_score = row.data[j]
        word_tfidf_mapping[i][word] = tfidf_score

# 'tfidf' 열에 단어와 해당 TF-IDF 가중치를 추가
#df['tfidf'] = [word_tfidf_mapping[i] for i in range(len(df))]

# TF-IDF 결과를 DataFrame 출력
df['han_tfidf_keywords'] = [list(word_tfidf_mapping[i].keys()) for i in range(len(df))]  # 각 행에 대한 키워드 리스트
df['han_tfidf_scores'] = [list(word_tfidf_mapping[i].values()) for i in range(len(df))]  # 각 행에 대한 키워드에 대한 점수 리스트

# TF-IDF 결과를 DataFrame 출력
print(df)



script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정
n_relative_path = "../../data/result/Hannanum+tfidf.csv"

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
n_file_path = os.path.join(script_dir, relative_path)

df.to_csv(n_file_path)

