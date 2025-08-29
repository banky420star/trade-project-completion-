FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=9000
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:9000/health || exit 1
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","9000","--workers","1"]
