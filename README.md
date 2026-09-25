# 아기 동요

파스텔 톤의 모바일·데스크톱 아기 동요 웹 앱입니다. 전래·퍼블릭 도메인 동요 30곡을 고르면 노래에 맞춰 한글·영어 가사가 따라갑니다. 한국 동요는 공개된 계이름과 박자에 맞춰 피아노로 다시 연주합니다.

## 실행

```bash
npm install
npm run dev
```

빌드:

```bash
npm run build
npm run preview
```

미리보기는 `http://127.0.0.1:4173` 입니다.

음원과 가사 타임코드를 다시 만들려면 (Python, numpy, ffmpeg):

```bash
python3 scripts/render_songs.py
```

## GitHub Codespaces에서 공개 페이지

이 저장소에는 `.devcontainer`가 있습니다. Codespace를 만들면 의존성을 설치하고 빌드한 뒤 4173 포트로 미리보기를 띄웁니다.

1. GitHub에서 이 브랜치로 Codespace를 만듭니다.
2. 포트 4173 공개 범위를 Public으로 바꿉니다. 터미널에서는 다음을 실행합니다.

```bash
gh codespace ports visibility 4173:public -c "$CODESPACE_NAME"
```

3. 브라우저 주소는 `https://<codespace이름>-4173.app.github.dev` 입니다. Codespace가 켜져 있는 동안만 접속됩니다.

## 구성

- 30곡 피아노 편곡 MP3 (`public/audio`)
- 박자에 맞춘 한글/영어 가사 (`src/data/songs.json`)
- 검색, 분류, 찜, 반복, 셔플, 수면 타이머
- 큰 터치 버튼과 파스텔 카드 UI
