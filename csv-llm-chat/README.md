## LLM 사용 허가 신청

1. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 에 접속하셔서 사용허가 신청을 진행합니다.
2. 신청이 허가되면 아래의 이미지 빌드 및 최초 실행부터 시작하여 주세요.

## 가상환경 설치 및 실행 순서

0. python과 pip이 없으실 경우, 3버전대의 python과 pip을 PC에 설치하여 주세요.
1. pip install uv
2. uv run jupyter lab (처음 실행 시 의존성 설치 때문에 매우 느립니다.)
3. console 에 http://localhost:8888/lab?token=... 이 나오면 컨트롤 + 마우스 왼쪽클릭하여 해당 url 에 접속합니다.
4. 파일탐색기에서 rag-chatbot-poc-v1.ipynb 을 찾아 클릭합니다.
5. 파일을 순서대로 실행하면서 마지막 셀까지 도달합니다.
6. 마지막 셀에서 Running on local URL : ... 에서 URL 부분을 클릭합니다.
7. 테스트합니다.

설치가 완료 되었으면, 다음부터 실행하실때에는 0번과 1번을 제외하고, 2번부터 진행하시면 됩니다.

