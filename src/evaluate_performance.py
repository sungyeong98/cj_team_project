# 1번 인덱스부터 유의미한 데이터 있음

import os
import csv

script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정
relative_path = "../data/result/hannanum_test_dataset.csv"

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
file_path = os.path.join(script_dir, relative_path)

# 빈 리스트를 생성하여 데이터를 저장할 준비
y_true = []
y_pred = []

# CSV 파일 열기
with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # 각 행을 반복하며 원하는 열의 데이터를 가져옴
    for row in csv_reader:
        if len(row) > -3:
            y_true.append(row[-3])
        if len(row) > -1:
            y_pred.append(row[-1])


# ------------------ Scikit-Learn: F1 score ------------------

from sklearn.metrics import f1_score

# F1 점수를 계산합니다.
f1 = f1_score(y_true, y_pred)

print("F1 Score:", f1)




# ------------------ NLTK: WordNet ------------------


'''
import nltk
nltk.download('wordnet')  # 예제 데이터 중 하나

# 추출된 키워드와 실제 키워드
extracted_keywords = ['keyword1', 'keyword2', ...]
actual_keywords = ['actual_keyword1', 'actual_keyword2', ...]

# 유사성 계산
similarities = []
for extracted_word in extracted_keywords:
    extracted_synsets = wordnet.synsets(extracted_word)
    for actual_word in actual_keywords:
        actual_synsets = wordnet.synsets(actual_word)
        if extracted_synsets and actual_synsets:
            similarity = extracted_synsets[0].path_similarity(actual_synsets[0])
            similarities.append(similarity)

# 유사성 평균 계산 또는 다른 메트릭 사용
average_similarity = sum(similarities) / len(similarities)

print(f"Average Word Similarity: {average_similarity}")
'''



'''

# ------------------ Gensim: 코사인 유사도 ------------------

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Gensim 모델 로드 또는 학습
word2vec_model = Word2Vec.load('word2vec_model.bin')

# 추출된 키워드와 실제 키워드 벡터 생성
extracted_keywords = ['keyword1', 'keyword2', ...]
actual_keywords = ['actual_keyword1', 'actual_keyword2', ...]

# 키워드 벡터 추출
extracted_keyword_vectors = [word2vec_model.wv[keyword] for keyword in extracted_keywords]
actual_keyword_vectors = [word2vec_model.wv[keyword] for keyword in actual_keywords]

# 코사인 유사성 측정
similarity_scores = cosine_similarity(extracted_keyword_vectors, actual_keyword_vectors)

# 유사성 평균 계산 또는 다른 메트릭 사용
average_similarity = similarity_scores.mean()

print(f"Average Cosine Similarity: {average_similarity}")

'''


