import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { RepeatMode, Song } from '../types'

const FAVORITES_KEY = 'baby-sing-favorites'

function loadFavorites(): string[] {
  try {
    const raw = localStorage.getItem(FAVORITES_KEY)
    return raw ? (JSON.parse(raw) as string[]) : []
  } catch {
    return []
  }
}

export function usePlayer(songs: Song[]) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [repeat, setRepeat] = useState<RepeatMode>('all')
  const [shuffle, setShuffle] = useState(false)
  const [favorites, setFavorites] = useState<string[]>(loadFavorites)
  const [open, setOpen] = useState(false)
  const [sleepUntil, setSleepUntil] = useState<number | null>(null)

  const current = songs.find((s) => s.id === currentId) ?? null
  const index = current ? songs.findIndex((s) => s.id === current.id) : -1

  const lyricIndex = useMemo(() => {
    if (!current) return -1
    const i = current.lyrics.findIndex((line) => time >= line.start && time < line.end)
    if (i >= 0) return i
    if (time >= (current.lyrics.at(-1)?.end ?? 0)) return current.lyrics.length - 1
    return 0
  }, [current, time])

  useEffect(() => {
    const audio = new Audio()
    audio.preload = 'auto'
    audioRef.current = audio
    const onTime = () => setTime(audio.currentTime)
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    return () => {
      audio.pause()
      audio.src = ''
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
    }
  }, [])

  const playSong = useCallback(
    (song: Song, autoplay = true) => {
      const audio = audioRef.current
      if (!audio) return
      setCurrentId(song.id)
      setTime(0)
      if (audio.src !== new URL(song.audio, window.location.href).href) {
        audio.src = song.audio
      }
      if (autoplay) void audio.play()
      setOpen(true)
    },
    [],
  )

  const toggle = useCallback(() => {
    const audio = audioRef.current
    if (!audio || !current) return
    if (audio.paused) void audio.play()
    else audio.pause()
  }, [current])

  const seek = useCallback((value: number) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = value
    setTime(value)
  }, [])

  const next = useCallback(
    (fromEnded = false) => {
      if (!songs.length) return
      if (fromEnded && repeat === 'one' && current) {
        seek(0)
        void audioRef.current?.play()
        return
      }
      if (fromEnded && repeat === 'off') {
        audioRef.current?.pause()
        return
      }
      let nextIndex = index + 1
      if (shuffle) {
        nextIndex = Math.floor(Math.random() * songs.length)
        if (songs.length > 1 && nextIndex === index) nextIndex = (index + 1) % songs.length
      } else if (nextIndex >= songs.length) {
        nextIndex = 0
      }
      playSong(songs[nextIndex])
    },
    [current, index, playSong, repeat, seek, shuffle, songs],
  )

  const prev = useCallback(() => {
    if (!songs.length) return
    if (time > 3 && current) {
      seek(0)
      return
    }
    const prevIndex = index <= 0 ? songs.length - 1 : index - 1
    playSong(songs[prevIndex])
  }, [current, index, playSong, seek, songs, time])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onEnded = () => next(true)
    audio.addEventListener('ended', onEnded)
    return () => audio.removeEventListener('ended', onEnded)
  }, [next])

  useEffect(() => {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites))
  }, [favorites])

  useEffect(() => {
    if (!sleepUntil) return
    const id = window.setInterval(() => {
      if (Date.now() >= sleepUntil) {
        audioRef.current?.pause()
        setSleepUntil(null)
      }
    }, 1000)
    return () => window.clearInterval(id)
  }, [sleepUntil])

  const toggleFavorite = useCallback((id: string) => {
    setFavorites((prevFav) =>
      prevFav.includes(id) ? prevFav.filter((x) => x !== id) : [...prevFav, id],
    )
  }, [])

  const cycleRepeat = useCallback(() => {
    setRepeat((mode) => (mode === 'all' ? 'one' : mode === 'one' ? 'off' : 'all'))
  }, [])

  return {
    current,
    playing,
    time,
    lyricIndex,
    repeat,
    shuffle,
    favorites,
    open,
    sleepUntil,
    setOpen,
    playSong,
    toggle,
    seek,
    next,
    prev,
    toggleFavorite,
    cycleRepeat,
    setShuffle,
    setSleepUntil,
  }
}
