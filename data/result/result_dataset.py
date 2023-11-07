import os 

from konlpy.tag import Hannanum
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

script_dir = os.path.dirname(__file__)
# 파일의 상대 경로를 지정
relative_path = "../../data/processed/hannanum_dataset.csv"

# 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
file_path = os.path.join(script_dir, relative_path)

# Excel 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(file_path)

