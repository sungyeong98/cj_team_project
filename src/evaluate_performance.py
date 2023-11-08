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
df5=pd.read_csv(file_path5)     #kobart+okt+yake

y_true=df1['imp_words'].astype(str).tolist()
y_pred1=df1['ex_words'].astype(str).tolist()
y_pred2=df2['ex_words'].astype(str).tolist()
y_pred3=df3['han_tfidf_keywords'].astype(str).tolist()
y_pred4=df4['okt_tfidf_keywords'].astype(str).tolist()
y_pred5=df5['YAKE'].astype(str).tolist()
# 빈 리스트를 생성하여 데이터를 저장할 준비
#y_true = []
#y_pred = []

# ------------------ Scikit-Learn: F1 score ------------------
from sklearn.metrics import f1_score

# F1 점수를 계산합니다.

f1_test1,f1_test2,f1_test3,f1_test4,f1_test5=0,0,0,0,0
for true_words,pred_words1,pred_words2,pred_words3,pred_words4,pred_words5 in zip(y_true,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5):
    true_word=true_words.split(',')
    pred_word1=pred_words1.split(',')
    pred_word2=pred_words2.split(',')
    pred_word3=','.join(pred_words3).split(',')
    pred_word4=','.join(pred_words4).split(',')
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

f1_test1=(f1_test1/len(y_true))
f1_test2=(f1_test2/len(y_true))
f1_test3=(f1_test3/len(y_true))
f1_test4=(f1_test4/len(y_true))
f1_test5=(f1_test5/len(y_true))

print("# ------------------ Scikit-Learn: F1 score ---------------- #")
print('한나눔+keybert : ',f1_test1)
print('okt+keybert : ',f1_test2)
print('한나눔+tfidf : ',"{:.10f}".format(f1_test3))
print('okt+tfidf : ',"{:.10f}".format(f1_test4))
print('kobart+YAKE : ',f1_test5)
print("# ---------------------------------------------------------- #")

# ------------------ Jaccard Similarity ------------------

# Jaccard Similarity : 두 집합 A와 B가 있을 때, 이 두 집합이 얼마나 유사한지를 알려주는 척도
jac_test1=0
jac_test2=0
jac_test3=0
jac_test4=0
jac_test5=0

for i in range(len(y_true)):
    words0 = set(y_true[i])
    words1 = set(y_pred1[i])
    words2 = set(y_pred2[i])
    words3 = set(y_pred3[i])
    words4 = set(y_pred4[i])
    words5 = set(y_pred5[i])

    intersection = words0.intersection(words1)
    union = words0.union(words1)
    jaccard_similarity = len(intersection) / len(union)
    jac_test1 = jac_test1 + jaccard_similarity 

    intersection = words0.intersection(words2)
    union = words0.union(words2)
    jaccard_similarity = len(intersection) / len(union)
    jac_test2 = jac_test2 + jaccard_similarity 

    intersection = words0.intersection(words3)
    union = words0.union(words3)
    jaccard_similarity = len(intersection) / len(union)
    jac_test3 = jac_test3 + jaccard_similarity 

    intersection = words0.intersection(words4)
    union = words0.union(words4)
    jaccard_similarity = len(intersection) / len(union)
    jac_test4 = jac_test4 + jaccard_similarity 

    intersection = words0.intersection(words5)
    union = words0.union(words5)
    jaccard_similarity = len(intersection) / len(union)
    jac_test5 = jac_test5 + jaccard_similarity 


jac_test1 = jac_test1 / len(y_true)
jac_test2 = jac_test2 / len(y_true)
jac_test3 = jac_test3 / len(y_true)
jac_test4 = jac_test4 / len(y_true)
jac_test5 = jac_test5 / len(y_true)

print("# --------------- Jaccard Similarity ----------------------- #")
print("한나눔+keybert :", jac_test1)
print("okt+keybert :", jac_test2)
print("한나눔+tfidf :", jac_test3)
print("okt+tfidf :", jac_test4)
print("kobart+YAKE :", jac_test5)
print("# ---------------------------------------------------------- #")

# ------------------ Gensim: 코사인 유사도 ------------------
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Gensim 모델 로드 또는 학습
tokens=[word.split(',') for word in y_true]
model=Word2Vec(tokens,min_count=1)
model_name='gensim_score_model'
model.save(model_name)

score_model=Word2Vec.load(model_name)

n_score1,n_score2,n_score3,n_score4,n_score5=0,0,0,0,0
for true_words,pred_words1,pred_words2,pred_words3,pred_words4,pred_words5 in zip(y_true,y_pred1,y_pred2,y_pred3,y_pred4,y_pred5):
    true_word=true_words.split(',')
    pred_word1=pred_words1.split(',')
    pred_word2=pred_words2.split(',')
    pred_word3=','.join(pred_words3).split(',')
    pred_word4=','.join(pred_words4).split(',')
    pred_word5=pred_words5.split(',')

    n_score1+=score_model.wv.n_similarity(true_word,pred_word1)
    n_score2+=score_model.wv.n_similarity(true_word,pred_word2)
    n_score3+=score_model.wv.n_similarity(true_word,pred_word3)
    n_score4+=score_model.wv.n_similarity(true_word,pred_word4)
    n_score5+=score_model.wv.n_similarity(true_word,pred_word5)
#print(score_model.wv.n_similarity(y_true[0].split(','),y_pred1[0].split(',')))
n_score1=(n_score1/len(y_true))
n_score2=(n_score2/len(y_true))
n_score3=(n_score3/len(y_true))
n_score4=(n_score4/len(y_true))
n_score5=(n_score5/len(y_true))
print("# ---------------- Gensim: 코사인 유사도 ------------------- #")
print('한나눔+keybert : ',n_score1)
print('okt+keybert : ',n_score2)
print('한나눔+tfidf : ',n_score3)
print('okt+tfidf : ',n_score4)
print('kobart+YAKE : ',n_score5)
print("# ---------------------------------------------------------- #")

