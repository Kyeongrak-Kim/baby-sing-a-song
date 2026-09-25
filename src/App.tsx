import { useEffect, useMemo, useState } from 'react'
import songsData from './data/songs.json'
import { usePlayer } from './hooks/usePlayer'
import type { Category, RepeatMode, Song } from './types'

const songs = songsData as Song[]

type FilterId = 'all' | 'favorites' | 'korean' | Category

const CATEGORIES: { id: FilterId; label: string }[] = [
  { id: 'all', label: '전체' },
  { id: 'favorites', label: '찜' },
  { id: 'korean', label: '우리 동요' },
  { id: 'lullaby', label: '자장가' },
  { id: 'animal', label: '동물' },
  { id: 'play', label: '놀이' },
  { id: 'story', label: '이야기' },
  { id: 'daily', label: '일상' },
]

function formatTime(sec: number) {
  const s = Math.max(0, Math.floor(sec))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

const REPEAT_LABEL: Record<RepeatMode, string> = {
  all: '전체 반복',
  one: '한 곡 반복',
  off: '반복 끄기',
}

export default function App() {
  const player = usePlayer(songs)
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<FilterId>('all')
  const [showSleep, setShowSleep] = useState(false)

  useEffect(() => {
    document.querySelector('.lyric.active')?.scrollIntoView({
      block: 'center',
      behavior: 'smooth',
    })
  }, [player.lyricIndex, player.current?.id])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return songs.filter((song) => {
      if (category === 'favorites' && !player.favorites.includes(song.id)) return false
      if (category === 'korean' && !song.korean) return false
      if (category !== 'all' && category !== 'favorites' && category !== 'korean' && song.category !== category)
        return false
      if (!q) return true
      return `${song.titleKo} ${song.titleEn}`.toLowerCase().includes(q)
    })
  }, [category, player.favorites, query])

  return (
    <div className="app">
      <header className="hero">
        <div>
          <h1>아기 동요</h1>
          <p>30곡을 고르면 노래에 맞춰 가사가 반짝여요</p>
        </div>
        <div className="moon" aria-hidden="true">
          🌙
        </div>
      </header>

      <input
        className="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="동요 이름 찾기"
        aria-label="동요 검색"
      />

      <div className="chips" role="tablist" aria-label="분류">
        {CATEGORIES.map((item) => (
          <button
            key={item.id}
            className={`chip ${category === item.id ? 'active' : ''}`}
            onClick={() => setCategory(item.id)}
            role="tab"
            aria-selected={category === item.id}
          >
            {item.label}
          </button>
        ))}
      </div>

      <section className="grid">
        {filtered.length === 0 && <div className="empty">아직 찜한 동요가 없어요</div>}
        {filtered.map((song) => {
          const active = player.current?.id === song.id
          return (
            <button
              key={song.id}
              className={`card ${active ? 'playing' : ''}`}
              style={{ background: song.color }}
              onClick={() => player.playSong(song)}
            >
              <span className="badge">{song.emoji}</span>
              <div>
                <h2>{song.titleKo}</h2>
                <div className="en">{song.titleEn}</div>
              </div>
              <div className="meta">
                <span>{formatTime(song.duration)}</span>
                <span>{active && player.playing ? '재생 중' : '듣기'}</span>
              </div>
            </button>
          )
        })}
      </section>

      {player.current && !player.open && (
        <div className="mini">
          <button
            className="emoji"
            style={{ background: player.current.color }}
            onClick={() => player.setOpen(true)}
            aria-label="재생 화면 열기"
          >
            {player.current.emoji}
          </button>
          <button className="titles" onClick={() => player.setOpen(true)}>
            <b>{player.current.titleKo}</b>
            <span>{player.current.lyrics[player.lyricIndex]?.ko ?? player.current.titleEn}</span>
          </button>
          <button className="icon-btn primary" onClick={player.toggle} aria-label="재생">
            {player.playing ? '❚❚' : '▶'}
          </button>
        </div>
      )}

      {player.current && player.open && (
        <div
          className="sheet"
          style={{ ['--sheet-bg' as string]: player.current.color }}
          role="dialog"
          aria-label={`${player.current.titleKo} 재생`}
        >
          <div className="sheet-top">
            <button className="icon-btn" onClick={() => player.setOpen(false)} aria-label="목록으로">
              ⌄
            </button>
            <button
              className="icon-btn"
              onClick={() => player.toggleFavorite(player.current!.id)}
              aria-label="찜하기"
            >
              {player.favorites.includes(player.current.id) ? '♥' : '♡'}
            </button>
          </div>

          <div className={`sheet-art ${player.playing ? '' : 'paused'}`}>{player.current.emoji}</div>
          <h2>{player.current.titleKo}</h2>
          <div className="en-title">{player.current.titleEn}</div>

          <div className="lyrics">
            {player.current.lyrics.map((line, i) => (
              <div key={`${line.start}-${i}`} className={`lyric ${i === player.lyricIndex ? 'active' : ''}`}>
                <div className="ko">{line.ko}</div>
                <div className="en">{line.en}</div>
              </div>
            ))}
          </div>

          <div className="progress">
            <span>{formatTime(player.time)}</span>
            <input
              type="range"
              min={0}
              max={player.current.duration}
              step={0.1}
              value={Math.min(player.time, player.current.duration)}
              onChange={(e) => player.seek(Number(e.target.value))}
              aria-label="재생 위치"
            />
            <span>{formatTime(player.current.duration)}</span>
          </div>

          <div className="controls">
            <button className="icon-btn" onClick={player.prev} aria-label="이전 곡">
              ⏮
            </button>
            <button className="icon-btn primary play" onClick={player.toggle} aria-label="재생">
              {player.playing ? '❚❚' : '▶'}
            </button>
            <button className="icon-btn" onClick={() => player.next()} aria-label="다음 곡">
              ⏭
            </button>
          </div>

          <div className="tools">
            <div className="row">
              <button className="chip" onClick={player.cycleRepeat}>
                {REPEAT_LABEL[player.repeat]}
              </button>
              <button
                className={`chip ${player.shuffle ? 'active' : ''}`}
                onClick={() => player.setShuffle((v) => !v)}
              >
                섞기
              </button>
            </div>
            <button className="chip" onClick={() => setShowSleep((v) => !v)}>
              수면 타이머
            </button>
          </div>

          {showSleep && (
            <div className="sleep-menu">
              {[5, 10, 20].map((min) => (
                <button
                  key={min}
                  className={player.sleepUntil && Math.round((player.sleepUntil - Date.now()) / 60000) === min ? 'on' : ''}
                  onClick={() => player.setSleepUntil(Date.now() + min * 60_000)}
                >
                  {min}분
                </button>
              ))}
              <button onClick={() => player.setSleepUntil(null)}>끄기</button>
            </div>
          )}

          <div className="credit">{player.current.source}</div>
        </div>
      )}
    </div>
  )
}
