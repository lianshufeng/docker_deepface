FROM python:3.11-bullseye

# Install dependencies
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    git zip unzip wget curl htop gcc \
#    && rm -rf /var/lib/apt/lists/*


# 安装 OpenCV 运行所需的底层库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
ADD ./src ./
ADD ./requirements.txt ./requirements.txt


RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]