# 아기 동요

파스텔 톤의 모바일·데스크톱 아기 동요 웹 앱입니다. 전래·퍼블릭 도메인 동요 30곡을 고르면 노래에 맞춰 한글·영어 가사가 따라갑니다.

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

음원과 가사 타임코드를 다시 만들려면:

```bash
python3 scripts/render_songs.py
```

## 구성

- 30곡 오르골 편곡 MP3 (`public/audio`)
- 박자에 맞춘 한글/영어 가사 (`src/data/songs.json`)
- 검색, 분류, 찜, 반복, 셔플, 수면 타이머
- 큰 터치 버튼과 파스텔 카드 UI
