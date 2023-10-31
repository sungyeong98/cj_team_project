# 베이스 이미지
FROM python:3.8

# 작업 디렉토리,-
WORKDIR /app

# 필요한 라이브러리 및 도구를 설치
RUN apt-get update && apt-get install -y \
    git \
    curl

# 프로젝트에 필요한 Python 패키지를 설치
# requirements.txt 파일을 사용하여 종속성을 정의
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 프로젝트 파일을 도커 이미지로 복사합니다.
COPY . .

# 컨테이너를 실행할 때의 명령어를 지정합니다. 이 부분은 프로젝트에 따라 다를 수 있습니다.
# CMD ["python", "main.py"]
CMD echo "container started"
