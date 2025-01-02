# rag-chatbot라는 이름을 가진 이미지와 컨테이너 확인
IMAGE_EXISTS=$(docker images -q rag-chatbot)
CONTAINER_EXISTS=$(docker ps -a -q --filter "ancestor=rag-chatbot")

# 이미지가 존재하면 빌드하지 않음
if [ -n "$IMAGE_EXISTS" ]; then
  echo "이미 rag-chatbot 이미지가 존재합니다. 빌드를 건너뜁니다."
else
  echo "rag-chatbot 이미지를 빌드합니다."
  docker build -t rag-chatbot .
fi

# 컨테이너가 존재하면 실행하지 않음
if [ -n "$CONTAINER_EXISTS" ]; then
  echo "이미 rag-chatbot 컨테이너가 존재합니다. run.sh 를 실행시켜 주세요."
else
  echo "rag-chatbot 컨테이너를 실행합니다."
  docker run -it --gpus=all -p 7860:7860 rag-chatbot
fi