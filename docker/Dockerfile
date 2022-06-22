FROM jjanzic/docker-python3-opencv:opencv-4.0.0
RUN pip install -U pip

WORKDIR /app
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

COPY src src
RUN mkdir log

COPY data/alaska /root/Datasets/ALASKA_v2_TIFF_256_COLOR
COPY data/boss /root/Datasets/BOSS_tiles
