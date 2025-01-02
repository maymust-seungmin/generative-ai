## .env file

1. RAG-docker 디렉토리 밑에 .env 파일을 생성합니다.
2. HF_TOKEN={사용자의 Hugging Face Token}
3. {사용자의 Hugging Face Token} 부분을 허깅페이스에서 발급받은 사용자의 토큰으로 치환합니다.
4. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct 에 접속하셔서 사용허가 신청을 진행합니다.
5. 신청이 허가되면 아래의 이미지 빌드 및 최초 실행부터 시작하여 주세요.

## 이미지 빌드 및 최초 실행

```bash build.sh``` 를 콘솔에 입력합니다.

## 이후 새로운 세션 실행 할 때마다

```bash run.sh``` 를 콘솔에 입력합니다.