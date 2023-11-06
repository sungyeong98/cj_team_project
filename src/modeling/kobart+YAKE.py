# pip install torch
# pip install transformers
# pip install yake
# pip install pandas
# pip install konlpy

# 가져오는 파일 이름 변경해야함

import os
import pandas as pd
from konlpy.tag import Okt
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import yake
import sys
# 현재 스크립트 파일의 디렉토리 경로를 얻음
script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정 & 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
relative_path = "../" 
file_path = os.path.join(script_dir, relative_path)
sys.path.append(file_path)
from data_preprocessing import preprocessing

#df=preprocessing.prepro()[:10]
df=preprocessing.prepro()

print(df)
# ----- kobart로 요약본 추출 ----- #
#  Load Model and Tokenize
tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")

# 요약 추출
df['summary']=''
for idx,row in df.iterrows():
    # Encode Input Text
    input_text = row['text']
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_length=142,
        min_length=56,
        num_beams=4,
    )
    # Decoding Text
    df.at[idx,'summary'] = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    print(df.at[idx,'summary'])

# ----- Okt로 명사 추출 ----- #
# Okt 객체 생성
okt = Okt()

# 명사 추출
df['noun']=''
for idx,row in df.iterrows():
    temp=okt.nouns(row['summary'])
    temp_list=[word for word in temp if len(word)>0]
    df.at[idx,'noun']=' '.join(temp_list)

# ----- YAKE로 키워드 추출 ----- #
# YAKE 객체를 생성하고 키워드 추출을 위한 옵션을 설정
kw_extractor = yake.KeywordExtractor(lan="ko", n=1, dedupLim=0.9, top=5, features=None)

df['YAKE']=''
df['YAKE_score']=''
for idx,row in df.iterrows():
    keywords = kw_extractor.extract_keywords(row['noun'])
    kw_temp = ''
    score_temp = ''
    for kw, score in keywords:
        print("키워드:", kw, "점수:", score)
        kw_temp = kw_temp + ',' + str(kw)
        score_temp = score_temp + ',' +str(score)
    df.at[idx,'YAKE']=kw_temp[1:]
    df.at[idx,'YAKE_score']=score_temp[1:]

# 새로운 csv 파일 경로 지정 및 생성
script_dir = os.path.dirname(__file__)
new_relative_path = "../../data/result/kobart_test_dataset.csv"
new_file_path = os.path.join(script_dir, new_relative_path)

# csv 파일 저장
df.to_csv(new_file_path)