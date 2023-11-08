import os
import pandas as pd
from konlpy.tag import Okt
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import yake
import sys

def make_blank(keyword):
    # 글자길이만큼의 빈칸을 만들어주는 함수
    word_length = len(keyword)
    blank = '('
    for i in range(word_length):
        blank = blank + '__'
    blank = blank + ')'
    return blank

def replace_keywords_with_placeholder(content, keywords):
    # keywords 리스트를 반복하면서 각 키워드를 '( )'로 대체해주는 함수
    for keyword in keywords:
        # 문자열의 replace 메서드를 사용하여 키워드를 '( )'로 대체
        #content = content.replace(keyword, '(         )')
        content = content.replace(keyword, make_blank(keyword))
    return content

def GetKeyword(input_text):
    # kobart
    # Load Model and Tokenize
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
    model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
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
    summary = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

    # okt
    # Okt 객체 생성
    okt = Okt()
    # 명사 추출
    temp=okt.nouns(summary)
    temp_list=[word for word in temp if len(word)>0]
    nouns=' '.join(temp_list)
    # yake
    kw_extractor = yake.KeywordExtractor(lan="ko", n=1, dedupLim=0.9, top=5, features=None)
    yake_kw = []
    yake_score = []
    keywords = kw_extractor.extract_keywords(nouns)
    for kw, score in keywords:
        #print("키워드:", kw, "점수:", score)
        yake_kw.append(kw)
        #yake_score.append(score)

    # 빈칸 생성
    result = replace_keywords_with_placeholder(input_text, yake_kw)

    return result

# 사용 예시
content = "이 가옥은 현재 소유자인 손정호(孫定鎬)의 5대조가 많은 재물을 들여 1780년경에 세웠다고 한다. 원래는 교롱암가나다라(敎聾庵)이라고 하였으나, 손정호의 4대조가 문과에 합격하여 궁내부(宮內府) 주사(主事)의 자리에 있었다고 하여 주사댁이라 불리고 있다. ‘ㄱ’자 모양의 안채와 사랑채를 왼편과 오른편에 나란히 두고 각각 맞담을 쌓아 별도의 안마당 공간을 이루었다. 안채는 이 지방 큰 집들과 공통되는 격식(格式)을 갖추고 있다. 안채와 사랑채의 앞면으로 튀어나온 부분에는 넓은 곡식 창고와 고방(庫房)을 두어 당시에 살림이 풍족하였던 것을 보여 주고 있다. 사랑채도 ‘ㄱ’자형 평면으로 정면 5칸, 측면 2칸이다. 안마당 공간과 따로 출입문을 두고 있는 점은 특이하다. 안마당 한가운데에 초가지붕의 방앗간, 외양간을 자리하게 하였고 사랑마당에는 동쪽으로 치우쳐 초당(草堂)을 배치하였다."
result = GetKeyword(content)
print(result)
