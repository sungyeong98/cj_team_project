import pandas as pd
import re
import os

def prepro():
    # 현재 스크립트 파일의 디렉토리 경로를 얻음
    script_dir = os.path.dirname(__file__)

    # 파일의 상대 경로를 지정
    relative_path = "../../data/raw/golden_dataset.csv" 

    # 상대 경로를 현재 스크립트 파일의 경로와 조합하여 파일 경로를 얻음
    file_path = os.path.join(script_dir, relative_path)

    # CSV 파일을 데이터프레임으로 읽어오기
    df = pd.read_csv(file_path,encoding='CP949')
    
    #필요없는 부분 제거
    df=df.drop(['번호','요약문'],axis=1)
 
    #열이름 수정
    df = df.rename(columns={'본문': 'text', '중요단어': 'imp_words'})
    #타입 수정
    df['imp_words'] = df['imp_words'].astype(str)


    # 모든 값에 대해 괄호와 그 안의 내용, 한자, 특수문자 삭제
    df = df.applymap(lambda x: re.sub(r'\([^)]*\)|[^가-힣ㄱ-ㅎㅏ-ㅣ\s,0-9]', '', x))

    # 결과 확인
    #print(df)
    return df
