# 위클리 챌린지 — Docker 컨테이너화 / AWS EC2 배포 / CI·CD 파이프라인

대상 프로젝트: [til-rag](https://github.com/KIMDONGWOOK12/til-rag) (TIL 기반 RAG QA 시스템)

---

## 먼저 자주 나오는 용어

자주 나오는 용어를 정리 해봤습니다.

| 용어 | 뜻 |
|---|---|
| Docker | 애플리 케이션과 그 실행에 필요한 모든 것(코드, 라이브러리, 설정)을 하나로 묶어, 어떤 컴퓨터에서든 동일하게 실행되도록 하는 오픈소스 플랫폼 |
| Dockerfile | Docker 이미지를 만들기 위한 명령어와 설정을 담은 텍스트 스크립트 파일. |
| Image | 애플리케이션 실행에 필요한 모든 것을 포함한, 읽기 전용(Read-only) 스냅샷 |
| Container | 이미지를 실제로 실행한 상태. 애플리케이션과 그 종속성이 격리되어 돌아가는 단위 |
| Docker Hub(Registry) | 컨테이너 이미지를 저장하고 배포하는 중앙 저장소 |
| Docker Compose | 여러 컨테이너로 구성된 애플리케이션 YAML 파일로 정의하고, 단일 명령어로 실행*관리하는 도구|
| Volume | 컨테이너 종료 후에도 데이터를 유지하고, 컨테이너 간 공유하기 위한 저장소 |
| EC2 | 필요할 때 가상 서버(인스턴스)를 만들어 사용할 수 있는 AWS의 클라우드 컴퓨팅 서비스
| Security Group | EC2 인스턴스에 대한 네트워크 트래픽을 제어하는 가상 방화벽 |
| SSH / 키페어(.pem) | 네트워크를 통해 원격 서버에 안전하게 접속하고 명령을 실행하는 암호화된 프로토콜. 키페어는 그 접속에 쓰이는 공개키 / 개인키 쌍 |
| Swap | 물리 메모리가 부족할 때 디스크 공간을 임시 메모리처럼 사용하는 영역 |
| Continuous Integration (CI) | 코드 변경 사항을 자동으로 빌드하고 테스트하여 검증하는 과정 |
| Continuous Deployment (CD) | 검증된 코드를 자동으로 실제 서버에 배포하는 과정 |
| GitHub Actions | GitHub 내에서 코드 빌드, 테스트, 배포 등의 워크플로우를 자동화하는 CI/CD 도구 |




## 챌린지 1 — 지금까지 구축한 FastAPI 서버를 Docker 컨테이너로 패키징하고 Docker Compose로 실행해보기

### 1-1. Dockerfile 작성

프로젝트를 어떻게 컨테이너로 포장할지 적는 설계도입니다.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

의존성 파일(`pyproject.toml`, `uv.lock`)을 먼저 복사하고 그 다음에 전체 코드(`COPY . .`)를
복사한 이유는 레이어 캐싱 때문입니다. 코드만 바뀌었을 때 라이브러리 설치 단계를 다시 하지
않아 재빌드가 빨라지기 때문 입니다.

### 1-2. docker-compose.yml 작성

도커를 어떻게 실행할지에 대한 설정.

```yaml
services:
  til-rag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_db:/app/chroma_db
    env_file:
      - .env
```

`volumes`로 `chroma_db`를 연결한 이유는, 컨테이너는 종료 시 내부 데이터가 사라지기 때문입니다. 
299개 청크가 들어있는 벡터DB는 컨테이너 밖(호스트)에 남아야 합니다.

### 1-3. .dockerignore 작성

`.env`, `chroma_db/`, `.venv/`, `.git/` 등을 이미지에서 제외.
API 키가 이미지에 박히는 것을 막고, 불필요한 레이어로 이미지가 커지는 것도 방지합니다.

### 1-4. 이미지 빌드 및 실행 확인

```bash
docker build -t til-rag:latest .
docker compose up
```

브라우저에서 `http://localhost:8000/docs` 접속 → `/ask` 질문 응답까지 확인

---

## 챌린지 2 — 컨테이너 이미지를 AWS EC2에 배포하고 외부에서 접근 가능하도록 구성하기

### 진행 순서

**1. 아키텍처 확인 및 재빌드**
```bash
docker image inspect til-rag:latest --format '{{.Os}}/{{.Architecture}}'
# → linux/arm64
docker build --platform linux/amd64 -t til-rag:latest .
```
맥북이 Apple Silicon(M칩)이라 처음 빌드한 이미지가 `arm64`로 만들어져 있었습니다.
EC2는 x86(amd64)이라 그대로 올리면 실행되지 않을 수 있어 `--platform` 옵션으로 재빌드했습니다.

**2. Docker Hub 로그인 확인** — 계정 `ehddnr8838`

**3. 이미지에 계정 이름표 붙이기**
```bash
docker tag til-rag:latest ehddnr8838/til-rag:latest
```

**4. Docker Hub에 업로드**
```bash
docker push ehddnr8838/til-rag:latest
```
3GB 가까운 이미지라 업로드 중 네트워크가 끊겨 `broken pipe`가 발생했었습니다.
재실행을 해보니 이미 올라간 레이어는 `Layer already exists`로 건너뛰고 남은 것만 올라가 성공했습니다.

**5. EC2 인스턴스 시작** — AWS 콘솔에서 중지 상태였던 `orbit_practice`(t3.micro)를 시작

**6. 보안 그룹 설정** — 인바운드 규칙에 8000번 포트(TCP, `0.0.0.0/0`) 추가

**7. SSH 접속**
```bash
chmod 400 EXPRESS-BE.pem
ssh -i EXPRESS-BE.pem ubuntu@<퍼블릭IP>
```
인스턴스를 프로젝트 실행하기 위할때만 켜두기에 매번 킬때마다 값이 달라서 퍼블릭 IP로 작성 해뒀씀다.

**8. EC2에 Docker 설치** — 공식 저장소 방식(GPG 키 추가 → 저장소 등록 → `docker-ce` 설치)

**9. 프로젝트 폴더 및 docker-compose.yml 작성 (EC2용)**
```yaml
services:
  til-rag:
    image: ehddnr8838/til-rag:latest   # ← 로컬은 build: . / EC2는 image:
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./til_notes:/app/til_notes
    env_file:
      - .env
```
로컬용과 결정적으로 다른 점은 `build: .` 대신 `image:`를 쓴다는 것입니다.
EC2에는 Dockerfile이 없으므로 직접 조립하지 않고 Docker Hub에서 완성품을 받아옵니다.

이 과정에서 `root` 계정으로 폴더를 만들어 `ubuntu` 계정이 접근하지 못하는 문제가 있었어서,
`chown -R ubuntu:ubuntu`로 소유자를 변경해 해결했습니다.

**10. 이미지 받기**
```bash
docker compose pull
```

**11. Git에 없는 파일 전송**
```bash
scp -i EXPRESS-BE.pem .env ubuntu@<IP>:/home/ubuntu/til-rag/
scp -i EXPRESS-BE.pem -r chroma_db ubuntu@<IP>:/home/ubuntu/til-rag/
scp -i EXPRESS-BE.pem -r til_notes ubuntu@<IP>:/home/ubuntu/til-rag/
```
`.env`(API 키), `chroma_db`(벡터DB), `til_notes`(원본 문서)는 `.gitignore`와
`.dockerignore`에 등록되어 있어 Git에도, 이미지 안에도 들어가지 않습니다. 직접 전송해야 했습니다.

**12. 컨테이너 실행**
```bash
docker compose up -d
```

**13. 외부 접근 및 기능 확인**
브라우저에서 `http://<EC2 퍼블릭IP>:8000/docs` 접속 성공!
`/ask` 질문 응답, 답변에 표시된 파일명 클릭 시 원본 문서 열림까지 확인 완료 했습니다.

---

## 챌린지 3 — GitHub Actions로 코드 푸시 시 자동 빌드·배포되는 CI/CD 파이프라인 구축

### 진행 순서

**1. Docker Hub 토큰 발급** — Read & Write 권한 포함 (Write 없으면 push 불가)

**2. GitHub Secrets 등록** (Settings → Secrets and variables → Actions)

| 이름 | 역할 |
|---|---|
| `DOCKER_USERNAME` | Docker Hub 계정 |
| `DOCKER_TOKEN` | Docker Hub 로그인 |
| `SSH_PRIVATE_KEY` | EC2 접속용 pem 키 내용 전체 |
| `SERVER_HOST` | EC2 퍼블릭 IP |
| `SERVER_USER` | EC2 계정명 (`ubuntu`) |

**3. 워크플로우 파일 작성** — `.github/workflows/ci-cd.yml`

Job을 두 개로 분리했습니다. 이유는 아래에 있습니다.
```yaml
on:
  push:
    branches: [main]

jobs:
  build-and-push-image:
    # Docker Hub 로그인 → --platform linux/amd64 빌드 → push

  deploy:
    needs: build-and-push-image     # 빌드가 성공해야만 실행
    # SSH 키 준비 → EC2 접속 → docker compose pull → docker compose up -d
```

`needs`로 순서를 강제한 이유는, 빌드가 실패했는데 배포까지 진행되면
서버가 깨진 상태로 갱신될 수 있기 때문입니다.

**4. 코드 push 및 자동 실행 확인**

Actions 탭에서 두 Job 모두 초록 체크 확인 후, 배포된 서버가 정상 동작하는지 재확인.

---

## 겪은 문제와 해결

대부분의 오류들은 AI를 통하여 검색하여 진행하였으며, 구글링도 하여 찾아 원일을 찾은 것도 있습니다.

| 문제 | 증상 | 원인 | 해결 |
|---|---|---|---|
| 아키텍처 불일치 | — | 맥북 M칩이라 arm64로 빌드됨 | `--platform linux/amd64` 재빌드 |
| Dockerfile 못 찾음 | `open Dockerfile: no such file` | 홈 디렉토리에서 build 실행 | 프로젝트 폴더로 이동 후 재실행 |
| push 거부 | `insufficient_scope` | Docker Hub 로그인 세션 만료 | `docker login` 재인증 |
| push 중단 | `broken pipe` | 3GB 업로드 중 네트워크 끊김 | 재실행 (완료 레이어는 스킵) |
| 외부 접속 거부 | 브라우저 "연결 거부" | 보안 그룹에 8000번 포트 없음 | 인바운드 규칙 추가 |
| scp 실패 | `Permission denied` | root로 만든 폴더라 ubuntu가 못 씀 | `chown -R ubuntu:ubuntu` |
| git push 거부 | `without workflow scope` | GitHub 토큰에 workflow 권한 없음 | 토큰 편집해서 scope 추가 |
| deploy 실패 | `error in libcrypto` | SSH 키 복사 시 줄바꿈 형식 깨짐 | `pbcopy`로 파일 내용 정확히 재복사 |
| Docker 권한 거부 | `permission denied ... docker.sock` | `ubuntu`가 `docker` 그룹에 미소속 | `sudo usermod -aG docker ubuntu` |
| **44분째 정체** | `Extracting 366B` 반복 | **메모리 1GB, available 46MB — 이미지 압축 해제 불가** | **swap 1GB 추가 → 73초 완료** |

### 가장 오래 붙잡았던 문제 — 메모리 부족

`deploy` Job이 44분 동안 끝나지 않았습니다. 처음에는 그냥 원래 오래 걸리는건가 보다 했지만 로그를 자세히 보니
같은 레이어를 `Extracting 152B → 156B → 161B`처럼 몇 바이트씩 아주 조금씩 진행하고 있었습니다.
이상하여 팀원에게 물어보니 몇분 밖에 안걸렸다는 점에 이상함이 들어 자세히 알아본 결과
멈춘 게 아니라 **극도로 느린 것**이었습니다.

EC2에서 `free -h`로 확인해보니:

```
              total   used   free  shared  buff/cache  available
Mem:          911Mi  864Mi   71Mi   2.8Mi       113Mi      46Mi
Swap:            0B     0B     0B
```

`available`이 46MB, `Swap`은 0B. 인스턴스가 t3.micro(메모리 1GB)인데 이미지는 3GB.
압축 해제할 메모리 여유가 아예 없어서 진행이 사실상 멈춘 상태였습니다.
디스크는 램보다 훨씬 느리기에, swap이 많이 사용된다면 전체 성능이 느려진다는 단점을 아지만
swap 1GB를 추가하고 다시 실행하니 **73초 만에 완료**됐습니다.

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # 재부팅 후에도 유지
```

### 추가로 알게 된 것 — EC2 퍼블릭 IP는 고정이 아니다

인스턴스를 중지하고 다시 시작하니 퍼블릭 IP가 바뀌었습니다.
IP가 바뀌면 GitHub Secrets의 `SERVER_HOST`도 같이 수정해야 CI/CD가 계속 동작 한다는 것 입니다.
고정 IP가 필요하다면 Elastic IP를 연결해야 한다는 것도 알게 되었지만, 경험하지 못한 것들을 해보는 것 보다는 주어진 주제에
먼저 해본 후 다가가는것을 철칙으로 생각하기에 그렇게 진행해왔습니다.

---

## 회고

### 배포가 무엇인지 알게 됐다

이번 챌린지에서 가장 크게 남은 건 **"배포"라는 개념 자체를 알게 됐다는 점**이였습니다.

솔직히 그전까지는 배포라는 게 GitHub에 코드를 올리는 것 정도로 생각었습니다.
`git push`하면 그게 곧 배포라고 막연히 여겼었습니다.
그런데 이번에 직접 해보면서 그 둘이 완전히 다른 일이라는 걸 알게 됐습니다.

GitHub에 코드를 올리는 건 **코드를 보관하는 것**이고,
누군가 실제로 그 서비스를 쓸 수 있게 만드는 건 **완전히 별개의 과정**이었습니다.
서버를 직접 키고, 포트를 열고, 이미지를 옮기고, 컨테이너를 실행하고,
그렇게 여러 단계를 다 지나야 비로소 주소를 입력했을 때 무언가가 응답한다는 걸
과정으로 직접 겪었습니다.

특히 인상 깊었던 건 챌린지 1번까지 끝냈을 때였습니다.. 그때 GitHub에 커밋도 다 했는데,
EC2에는 아무 변화가 없었다. 코드를 올리는 것과 서버가 갱신되는 것이
전혀 연결되어 있지 않다는 걸 그때 눈으로 확인했습니다.
그리고 3번 챌린지에서 만든 GitHub Actions가 정확히 그 끊어진 두 지점을
이어주는 다리였다는 것도 이해가 됐던 점 입니다.

### 왜 Docker가 필요한지도 몸으로 알게 됐다

Docker도 처음엔 "왜 이렇게 복잡하게 포장을 해야 하지?", "포장이라는 개념이 왜 있는거지?" 싶었습니다.
그런데 EC2에 처음 접속했을 때, 그 컴퓨터에는 Python도 uv도 등등
아무것도 깔려 있지 않았다는걸 알 수 있었습니다.
코드 파일만 복사해봤자 실행될 수가 없는 환경이었습니다.

"코드와 그 코드가 필요로 하는 환경 전체를 통째로 묶어서 옮긴다"는 게
왜 필요한 일인지, 그 상황에 실제로 놓여봐야 이해되는 거였습니다.
아키텍처 문제(arm64 vs amd64)로 재빌드해야 했던 것도 같은 맥락
"어디서든 똑같이 돌아간다"는 말이 그냥 되는 게 아니라,
플랫폼까지 맞춰줘야 성립하는 조건부 약속이라는 걸 알게 됐습니다.

### 에러를 읽는 게 문제 해결의 절반이었다

이번 챌린지에서 겪은 문제가 열 개? 쯤 되는데, 돌아보면 대부분
**에러 메시지 안에 답이 이미 들어있었습니다.**

- `without workflow scope` → 권한이 없다고 명시해줌
- `error in libcrypto` → 키 자체를 못 읽는다는 뜻
- `permission denied ... docker.sock` → docker 접근 권한 문제
- `Extracting 366B` → 멈춘 게 아니라 느린 것

특히 44분 정체 건이 그랬었습니다. 처음엔 "멈췄다"고 판단하고 취소하려 했는데,
로그를 자세히 보니 아주 조금씩은 움직이고 있었습니다
그 차이를 알아보고 나서야 "느리다 = 자원이 부족하다"로 원인을 좁힐 수 있었고,
`free -h` 한 번으로 available 46MB를 확인해 바로 해결로 이어졌습니다.

이전에 교재에서 "메모리 부족 시 swap으로 완충한다"는 내용을 읽은 적이 있었는데,
그때는 그냥 지식이었습니다. 어디서 봤다 생각이 들어 다시 교재를 확인 해보니 맞았습니다.
이번에 직접 겪고 나니 왜 그런 조치가 필요한지가 완전히 다르게 이해됐습니다.

### 느낀점

지금까지의 커리큘럼을 따라오는 방식에서 왜이리 가져가는게 없지? 부족한 부분을 다시 다시 봐도
왜이리 못가져가지? 머리가 나쁘다는걸 알았지만 이렇게 안좋았나? 라는 생각을 계속해서 해왔습니다.

어느덧 11주차 위클리챌린지까지 완성을 하면서 다시 한번 돌이켜보며
나는 누구이지? 나는 잘 하고 있나? 내가 가져가고 있는게 있나? 라고 생각을 해보면
완전히 다 가져왔다 라는 생각은 안들지만 어느정도 아 RAG는 이거였지? 
LangGraph는 이거였지? 등.. 하며 하나씩 머리속에 한 부분씩 잡혀 있는 모습을 볼 수 있었습니다.

온라인때도 그렇고 오프라인도 그렇고 어느 부분에서 내가 그렇게 성장을 조금씩 할 수 있었을까? 라는 생각에서
저에게 크게 성장의 발판이 된 부분은 크게 3가지 였습니다.
- 1. 지속적으로 몰라도 포기하지않고 또 다시 찾아보는 습관
이 습관을 통해 2,3번 더 나아가 모를 때 마다 찾아보는 습관을 가지며 조급씩 습득하는 습관이 저에게 큰 힘이 된거 같습니다.
- 2. AI에 대한 거부감
AI에 대해 의존도를 낮춰야한다 낮춰야한다 하며 애써 외면을 해오던 제 자신에서 
AI는 저의 최고의 동료이자 친구이자 선생이자 멘토라는 인식을 가지며 하나하나 모르는것을 물어보며 같이 공부를 한다 생각을 갖고
나아가다 보니 하나씩 학습되는 모습이 보였습니다.
- 3. 시간 투자
기존에는 그저 앉아 있는 시간을 많이 갖는게 중요하다 생각을 하여 앉아있는 연습을 해왓습니다. 그치만 이제 앉는 시간을
투자 하는 방법을 알았습니다. 출퇴근 시간이 길다보니 그 시간이 너무 아까워 어떻게 하지? 하고 처음에는 여기서 받은
교재를 봤었습니다. 하지만 책 울렁증과 멀미가 겹쳐 너무 읽기에 쉽지 않았습니다. 두번 째는 유튜브 시청이였습니다.
유튜브로 당시에 배운 RAG부터 LangGraph까지 하나씩 관련 유튜브를 시청하며, 아 이거 강의시간에 들은거 같아
아 이건 배웠어 하며 하나씩 기억을 해내며 추가적인 학습을 이어왔습니다. 또한 집에가서도 꾸준히 공부와 복습을 하며
하나씩 뭘 해야지 하면서 계획을 잡고 나아가니 하나씩 눈에 들어온다는 점 입니다.

물론 지금도 당연히 완벽하다. 나 이제 좀 안다 라고 말을 할 순 없습니다. 그 부분은 당연하다 생각합니다. 하지만
이 지금의 방식을 통해 쭉 이어진다면 추후 완벽하지는 못해도 나 그래도 좀 안다 라고 말을 할 수 있을거 같다는
생각이 듭니다. 부족하지만 계속해서 채우려고 노력하며, 부족한것이 부끄러운것이 아닌 당연함이라 생각을 하여
회피하지 않고 앞으로 열심히 헤엄쳐 나갈 것 입니다.

### 앞으로의 계획

- 자동화된 테스트가 없어서 CI 단계에 test Job을 넣지 못했습니다
  테스트 코드를 작성하는 것이 다음 과제로 둬야겠다 생각 합니다.
- EC2 인스턴스 사양이 낮아 swap에 의존하고 있습니다.
  실제 운영이라면 인스턴스 사양을 올리는 게 근본 해결책일 것인거 같습니다. (팀프로젝트)
- Elastic IP를 연결하지 않아 인스턴스를 재시작할 때마다 IP가 바뀝니다.
  Secrets를 매번 수정하는 번거로움이 있어 인스턴스를 계속 켜둘지 지금 방법으로 계속
  할지 고민 해보고 정해야 겠습니다.