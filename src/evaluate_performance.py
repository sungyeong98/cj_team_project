# 1번 인덱스부터 유의미한 데이터 있음

import pandas as pd
import os
import csv

script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정
relative_path1 = "../data/result/hannanum_test_dataset.csv"
relative_path2 = "../data/result/okt_test_dataset.csv"
relative_path3 = "../data/result/hannanum_tfidf.csv"
relative_path4 = "../data/result/okt_tfidf.csv"
relative_path5 = "../data/result/kobart_dataset.csv"

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
file_path1 = os.path.join(script_dir, relative_path1)
file_path2 = os.path.join(script_dir, relative_path2)
file_path3 = os.path.join(script_dir, relative_path3)
file_path4 = os.path.join(script_dir, relative_path4)
file_path5 = os.path.join(script_dir, relative_path5)

df1=pd.read_csv(file_path1)     #한나눔+keybert
df2=pd.read_csv(file_path2)     #okt+keybert
df3=pd.read_csv(file_path3)     #한나눔+tfidf
df4=pd.read_csv(file_path4)     #okt+tfidf
df5=pd.read_csv(file_path5)     #kobart

y_true=df1['imp_words'].astype(str).tolist()
y_pred1=df1['ex_words'].astype(str).tolist()
y_pred2=df2['ex_words'].astype(str).tolist()
y_pred3=df3['han_tfidf_keywords'].astype(str).tolist()
y_pred4=df4['okt_tfidf_keywords'].astype(str).tolist()
y_pred5=df5['YAKE'].astype(str).tolist()
# 빈 리스트를 생성하여 데이터를 저장할 준비
#y_true = []
#y_pred = []

# CSV 파일 열기
'''
with open(file_path1, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # 각 행을 반복하며 원하는 열의 데이터를 가져옴
    for row in csv_reader:
        if len(row) > -3:
            y_true.append(row[-3])
        if len(row) > -1:
            y_pred.append(row[-1])

y_true=y_true[1:]
y_pred=y_pred[1:]
'''
# ------------------ Scikit-Learn: F1 score ------------------

from sklearn.metrics import f1_score

# F1 점수를 계산합니다.

f1_test1,f1_test2,f1_test3,f1_test4,f1_test5=0,0,0,0,0
for true_words,pred_words1,pred_words2,pred_words3,pred_words4,pred_words5 in zip(y_true,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5):
    true_word=true_words.split(',')
    pred_word1=pred_words1.split(',')
    pred_word2=pred_words2.split(',')
    pred_word3=pred_words3.split(',')
    pred_word4=pred_words4.split(',')
    pred_word5=pred_words5.split(',')

    common1=set(true_word)&set(pred_word1)
    common2=set(true_word)&set(pred_word2)
    common3=set(true_word)&set(pred_word3)
    common4=set(true_word)&set(pred_word4)
    common5=set(true_word)&set(pred_word5)

    precision1=len(common1)/len(pred_word1) if len(pred_word1)>0 else 0
    precision2=len(common2)/len(pred_word2) if len(pred_word2)>0 else 0
    precision3=len(common3)/len(pred_word3) if len(pred_word3)>0 else 0
    precision4=len(common4)/len(pred_word4) if len(pred_word4)>0 else 0
    precision5=len(common5)/len(pred_word5) if len(pred_word5)>0 else 0
    
    recall1=len(common1)/len(true_word) if len(true_word)>0 else 0
    recall2=len(common2)/len(true_word) if len(true_word)>0 else 0
    recall3=len(common3)/len(true_word) if len(true_word)>0 else 0
    recall4=len(common4)/len(true_word) if len(true_word)>0 else 0
    recall5=len(common5)/len(true_word) if len(true_word)>0 else 0

    if precision1+recall1>0:
        f1_test1+=(2*(precision1*recall1)/(precision1+recall1))
    if precision2+recall2>0:
        f1_test2+=(2*(precision2*recall2)/(precision2+recall2))
    if precision3+recall3>0:
        f1_test3+=(2*(precision3*recall3)/(precision3+recall3))
    if precision4+recall4>0:
        f1_test4+=(2*(precision4*recall4)/(precision4+recall4))
    if precision5+recall5>0:
        f1_test5+=(2*(precision5*recall5)/(precision5+recall5))

f1_test1=(f1_test1/len(y_true))*100
f1_test2=(f1_test2/len(y_true))*100
f1_test3=(f1_test3/len(y_true))*100
f1_test4=(f1_test4/len(y_true))*100
f1_test5=(f1_test5/len(y_true))*100
print('한나눔+keybert : ',f1_test1)
print('okt+keybert : ',f1_test2)
print('한나눔+tfidf : ',f1_test3)
print('okt+tfidf : ',f1_test4)
print('kobart : ',f1_test5)

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


