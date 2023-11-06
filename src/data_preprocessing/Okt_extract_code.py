import os
import pandas as pd
from konlpy.tag import Okt

import preprocessing
df=preprocessing.prepro()

# Okt 객체 생성
okt = Okt()

# 명사 추출
df['okt_nouns']=''
for idx,row in df.iterrows():
    temp=okt.nouns(row['text'])
    temp_list=[word for word in temp if len(word)>0]
    df.at[idx,'okt_nouns']=','.join(temp_list)

# csv 파일 생성 및 저장
script_dir = os.path.dirname(__file__)
new_relative_path = "../../data/processed/Okt_dataset.csv"
new_file_path = os.path.join(script_dir, new_relative_path)

df.to_csv(new_file_path,encoding='utf-8')
