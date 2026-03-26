# 1. 베이스 이미지
FROM python:3.10
# 2. 작업 디렉토리
WORKDIR /letter
# 3. 의존성 먼저 복사
COPY letter/requirements.txt .
# 4. 설치
RUN pip install --no-cache-dir -r requirements.txt
# 5. 전체 코드 복사
COPY letter/ .
# 6. 포트
EXPOSE 8000
# 7. 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]