from flask import Flask, request, render_template
import os, sys

#script_dir = os.path.dirname(__file__)
#relative_path = "/" 
#file_path = os.path.join(script_dir, relative_path)
#sys.path.append(file_path)
#import get_keyword

app = Flask(__name__)

@app.route("/")
def hello():
	return render_template('index.html')

@app.route('/keyword', methods=['GET', 'POST'])
def index():
    user_input = request.form.get('user_input')

    script_dir = os.path.dirname(__file__)
    relative_path = "/" 
    file_path = os.path.join(script_dir, relative_path)
    sys.path.append(file_path)
    import get_keyword
    result = get_keyword.GetKeyword(user_input)
    return render_template('page.html', message=result)
    #return f'결과: {result}'

if __name__ == '__main__':
    app.run(debug=True)