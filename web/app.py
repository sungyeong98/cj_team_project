from flask import Flask, request, render_template
#import os, sys
from konlpy.tag import Okt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 웹 폼에서 POST 요청을 처리하는 부분
        user_input = request.form['user_input']

        # 모델 수행 부분 => 함수로
        result = ''

        okt = Okt()
        noun=okt.nouns(user_input)
        for word in noun:
            result = result + ',' + str(word)

        return f'명사추출 결과: {result[1:]}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)