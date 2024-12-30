# rag_docker라는 이름을 가진 컨테이너 ID를 가져옴
CONTAINER_ID=$(docker ps -a -q --filter "ancestor=rag_docker")

# 컨테이너 ID가 존재하는지 확인
if [ -z "$CONTAINER_ID" ]; then
  echo "rag_docker 컨테이너를 찾을 수 없습니다. 확인 후 다시 시도하세요."
  exit 1
fi

# 컨테이너를 시작하고 attach
echo "rag_docker 컨테이너를 시작합니다: $CONTAINER_ID"
docker start -ai "$CONTAINER_ID"
