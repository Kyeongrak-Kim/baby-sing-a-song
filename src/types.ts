export type Category = 'lullaby' | 'animal' | 'play' | 'story' | 'daily'

export type LyricLine = {
  start: number
  end: number
  ko: string
  en: string
}

export type Song = {
  id: string
  no: number
  titleKo: string
  titleEn: string
  emoji: string
  category: Category
  korean?: boolean
  color: string
  accent: string
  bpm: number
  audio: string
  duration: number
  lyrics: LyricLine[]
  source: string
}

export type RepeatMode = 'all' | 'one' | 'off'
