
def make_blank(keyword):
    # 글자길이만큼의 빈칸을 만들어주는 함수
    word_length = len(keyword)
    blank = '('
    for i in range(word_length):
        blank = blank + ' '
    blank = blank + ')'
    return blank

def replace_keywords_with_placeholder(content, keywords):
    # keywords 리스트를 반복하면서 각 키워드를 '( )'로 대체해주는 함수
    for keyword in keywords:
        # 문자열의 replace 메서드를 사용하여 키워드를 '( )'로 대체
        #content = content.replace(keyword, '(         )')
        content = content.replace(keyword, make_blank(keyword))
    return content

# 사용 예시
content = "이 가옥은 현재 소유자인 손정호(孫定鎬)의 5대조가 많은 재물을 들여 1780년경에 세웠다고 한다. 원래는 교롱암가나다라(敎聾庵)이라고 하였으나, 손정호의 4대조가 문과에 합격하여 궁내부(宮內府) 주사(主事)의 자리에 있었다고 하여 주사댁이라 불리고 있다. ‘ㄱ’자 모양의 안채와 사랑채를 왼편과 오른편에 나란히 두고 각각 맞담을 쌓아 별도의 안마당 공간을 이루었다. 안채는 이 지방 큰 집들과 공통되는 격식(格式)을 갖추고 있다. 안채와 사랑채의 앞면으로 튀어나온 부분에는 넓은 곡식 창고와 고방(庫房)을 두어 당시에 살림이 풍족하였던 것을 보여 주고 있다. 사랑채도 ‘ㄱ’자형 평면으로 정면 5칸, 측면 2칸이다. 안마당 공간과 따로 출입문을 두고 있는 점은 특이하다. 안마당 한가운데에 초가지붕의 방앗간, 외양간을 자리하게 하였고 사랑마당에는 동쪽으로 치우쳐 초당(草堂)을 배치하였다."
keywords = ["손정호","5대조","1780년","주사댁","교롱암가나다라","궁내부","주사"]
result = replace_keywords_with_placeholder(content, keywords)
print(result)