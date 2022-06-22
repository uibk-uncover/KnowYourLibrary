FROM jjanzic/docker-python3-opencv:opencv-4.0.0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . src/
RUN mkdir log

ENTRYPOINT ["python", "run.py"]