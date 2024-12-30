## 가상환경 설치 및 실행 순서

0. python과 pip이 없으실 경우, 3버전대의 python과 pip을 PC에 설치하여 주세요.
1. pip install uv
2. uv run jupyter lab (처음 실행 시 의존성 설치 때문에 매우 느립니다.)
3. console 에 http://localhost:8888/lab?token=... 이 나오면 컨트롤 + 마우스 왼쪽클릭하여 해당 url 에 접속합니다.
4. 파일탐색기에서 rag-chatbot-poc-v1.ipynb 을 찾아 클릭합니다.
5. 좌측 상단의 run -> Run All Cells 를 클릭합니다. (전부 실행하는데 오래 걸립니다.)
6. 마지막 셀에서 Running on local URL : ... 에서 URL 부분을 클릭합니다.
7. 테스트합니다.

설치가 완료 되었으면, 다음부터 실행하실때에는 0번과 1번을 제외하고, 2번부터 진행하시면 됩니다.