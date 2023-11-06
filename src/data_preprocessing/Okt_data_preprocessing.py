# pip install openpyxl
# pip install pandas
# pip install konlpy

import os
import pandas as pd
from konlpy.tag import Okt

# 현재 스크립트 파일의 디렉토리 경로를 얻음
script_dir = os.path.dirname(__file__)

# 파일의 상대 경로를 지정
relative_path = "../../data/raw/golden_dataset.xlsx" 

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
file_path = os.path.join(script_dir, relative_path)

# Excel 파일을 데이터프레임으로 읽어오기
df = pd.read_excel(file_path)

# Okt 객체 생성
okt = Okt()

# 명사 추출
df['추출단어']=''
for idx,row in df.iterrows():
    temp=okt.nouns(row['본문'])
    temp_list=[word for word in temp if len(word)>0]
    df.at[idx,'추출단어']=','.join(temp_list)

# csv 파일 생성 및 저장
new_relative_path = "../../data/processed/Okt_dataset.csv"
new_file_path = os.path.join(script_dir, new_relative_path)

df.to_csv(new_file_path)