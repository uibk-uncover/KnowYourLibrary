FROM jjanzic/docker-python3-opencv:opencv-4.0.0
RUN pip install -U pip

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . src/
RUN mkdir log
