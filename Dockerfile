FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# EXPOSE 7860
# CMD ["flask", "app.py", "--host", "0.0.0.0", "--port", "7860"]
EXPOSE $PORT
CMD gunicorn --bind 0.0.0.0:$PORT app:app