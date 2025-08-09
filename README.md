# 블로그 원고 분석기 (Blog Post Analyzer)

이 프로젝트는 블로그 마케팅 원고 텍스트 파일을 분석하여 다양한 정보를 추출하고, 콘텐츠 생성을 돕는 파이썬 CLI 도구입니다.

---

## 주요 기능

1.  **단어 토큰화**: 텍스트 파일들에서 한글 두 글자 이상의 단어를 추출하여 출력합니다.
2.  **문장 분리**: 텍스트 파일들의 문장을 정규표현식 기반으로 분리하여 출력합니다.
3.  **AI 개체 인식 및 그룹화**: 여러 원고의 전체 내용을 분석하여, AI가 직접 핵심 개체(장소, 제품명, 메뉴 등)를 인식하고 의미적으로 유사한 항목끼리 그룹화하여 대표 키워드를 부여합니다.
4.  **표현 라이브러리**: AI를 사용하여 원고에서 특정 중분류(예: 긍정적 평가, 제품 특징)에 적합한 단어 또는 짧은 문장(표현)을 추출하여 `키: [밸류]` 형태의 라이브러리를 구축합니다.
5.  **카테고리별 문장 라이브러리**: 파일명을 카테고리로 삼아, 카테고리별로 문장을 수집하고 라이브러리를 구축합니다.
6.  **템플릿 생성**: AI가 추출한 파라미터 맵을 기반으로, 원고 내의 파라미터가 포함된 문장들을 AI에 보내 해당 값들을 `[대표 키워드]` 형태로 대체하여 템플릿을 생성합니다. 이를 통해 재사용 가능한 원고 구조를 만들 수 있습니다.

---

## 설치 및 사용법

### 1. 설치

프로젝트를 로컬에 설치하여 `blog-analyzer` 명령어를 사용할 수 있습니다.

```bash
# 프로젝트 디렉토리에서 다음 명령어를 실행합니다.
# 가상환경을 사용하고 있다면 먼저 활성화해주세요.
source venv/bin/activate

uvicorn main:app --reload

# pip를 사용하여 설치합니다.
pip install .
```

### 2. CLI 명령어 사용법

설치가 완료되면, 터미널에서 `blog-analyzer` 명령어를 사용할 수 있습니다.

```bash
# 사용 가능한 모든 명령어 확인
blog-analyzer --help
```

#### 원고 분석

`data` 폴더의 텍스트 파일을 분석하고 결과를 MongoDB에 저장합니다.

```bash
blog-analyzer analyze
```

#### 원고 생성

MongoDB의 최신 분석 데이터를 기반으로 원고를 생성합니다.

```bash
blog-analyzer generate --keywords "키워드1, 키워드2" --user-instructions "추가 지침"
```

#### API 서버 실행

FastAPI 기반의 웹 서버를 실행합니다.

```bash
blog-analyzer serve
```

---

## 개발 환경 설정 (기존 방식)

CLI를 직접 수정하거나 개발에 참여하고 싶다면 아래의 기존 방식을 따를 수 있습니다.

### 1. 프로젝트 복제 및 이동

```bash
git clone <저장소_URL>
cd blog_analyzer
```

### 2. 가상 환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 4. OpenAI API 키 설정

`.env` 파일에 `OPENAI_API_KEY`를 설정합니다.# ghost-store-generator-server
