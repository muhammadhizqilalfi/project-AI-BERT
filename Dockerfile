FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860


CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
