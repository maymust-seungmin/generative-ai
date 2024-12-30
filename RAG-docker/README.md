## 도커 설치 순서 가이드

1. 도커를 설치합니다. https://docs.docker.com/engine/install/ubuntu/
2. NVIDIA Docker 및 드라이버 설치 확인

    호스트 머신에 NVIDIA 드라이버가 설치되어 있는지 확인:

    ```nvidia-smi```

    정상적으로 출력되지 않으면 NVIDIA 드라이버 다운로드 페이지에서 적절한 드라이버를 설치하세요.
    nvidia-container-toolkit이 설치되어 있는지 확인:

    ```dpkg -l | grep nvidia-container-toolkit```

    설치되어 있지 않다면 다음 명령으로 설치하세요:
    
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
    && sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit
    ```
    설치 후 Docker를 재시작합니다:

    ```sudo systemctl restart docker```

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