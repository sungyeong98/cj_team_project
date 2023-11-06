from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
import webbrowser

# HTML 파일 이름
html_file = "index.html"

# 웹 페이지를 열기 위해 브라우저를 자동으로 실행
webbrowser.open("http://localhost:8000/" + html_file)

# 웹 서버 생성 및 시작
with HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler) as server:
    server.serve_forever()
