FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn openai openenv
CMD ["uvicorn","inference:app","--host","0.0.0.0","--port","7860"]
