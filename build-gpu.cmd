:: docker build ./ -f Dockerfile --build-arg HTTP_PROXY=http://192.168.0.54:1080 --build-arg HTTPS_PROXY=http://192.168.0.54:1080 -t lianshufeng/deepface

docker build ./ -f Dockerfile-gpu -t lianshufeng/deepface:gpu 