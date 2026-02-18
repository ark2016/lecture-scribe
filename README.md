# Lecture Scribe

CLI-инструмент на Python, который превращает видеозаписи лекций и аудиозаписи (диктофон) в структурированные конспекты с LaTeX-формулами.

- **Видео** — записи экрана из Telegram-звонков → Markdown + HTML с таймстемпами
- **Аудио** — диктофонные записи (.aac, .mp3, .m4a и др.) → Obsidian Markdown с callout-блоками

> Проект использует **бесплатную исследовательскую лицензию Mistral API** (free research tier). Все API-вызовы — бесплатны в рамках этого тарифа.

## Как это работает

### Режим видео

```
video.mp4
│
├── [Audio Pipeline]
│   ffmpeg → WAV → Voxtral Transcribe → transcript.json
│
├── [Visual Pipeline]
│   OpenCV (2 fps) → SSIM keyframe detection → Mistral OCR → LLM structuring → visual.json
│
└── [Merge]
    transcript + visual → synchronized blocks → output.md + output.html
```

**Три этапа:**

1. **Аудио** — ffmpeg извлекает звук, Voxtral Mini транскрибирует с таймстемпами
2. **Визуал** — OpenCV семплирует кадры, SSIM детектирует ключевые кадры, Mistral OCR читает текст, LLM структурирует в JSON с LaTeX
3. **Объединение** — синхронизация по таймстемпам, вывод в Markdown + HTML (с MathJax)

### Режим аудио (диктофон)

```
recording.aac
│
├── [ffmpeg] → WAV (16 kHz, mono)
├── [Voxtral] → транскрипт с сегментами
└── [LLM постобработка] → структурированный Obsidian Markdown
```

**Два этапа:**

1. **Транскрипция** — ffmpeg конвертирует аудио, Voxtral Mini транскрибирует
2. **LLM постобработка** — Mistral Large превращает сырой транскрипт в структурированные заметки с формулами (`$`, `$$`), callout-блоками Obsidian (`> [!definition]`, `> [!theorem]`, `> [!example]` и т.д.)

Режим определяется автоматически по расширению файла.

## Используемые модели Mistral

| Задача | Модель | Тип |
|--------|--------|-----|
| Транскрипция аудио | `voxtral-mini-latest` | Audio |
| OCR кадров | `mistral-ocr-latest` | OCR |
| Структурирование текста | `mistral-large-latest` | Chat |
| Постобработка транскрипта (аудио) | `mistral-large-latest` | Chat |

Все модели доступны на бесплатном исследовательском тарифе Mistral с ограничением по rate limit (~1 RPS).

## Установка

**Требования:**
- Python 3.11+
- ffmpeg (установлен в системе и доступен в PATH)

```bash
git clone https://github.com/<your-user>/lecture-scribe.git
cd lecture-scribe
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -e .
# или с фиксированными версиями:
pip install -r requirements.txt
```

**Настройка API-ключа:**

```bash
cp .env.example .env
# Отредактировать .env, вставить свой ключ:
# MISTRAL_API_KEY=your_key_here
```

Ключ можно получить на [console.mistral.ai](https://console.mistral.ai/) — бесплатный research tier.

## Использование

### Аудио (диктофон → Obsidian)

```bash
# Базовый запуск с указанием предмета
lecture-transcribe recording.aac --subject "Математический анализ" -o notes.md

# С кэшированием транскрипта (для повторных запусков LLM)
lecture-transcribe recording.aac --subject "Моделирование" -o notes.md \
  --save-transcript transcript.json --cache

# С терминами для улучшения транскрипции
lecture-transcribe recording.aac --subject "Линейная алгебра" -o notes.md \
  --terms "определитель,матрица,собственное значение"
```

### Видео (запись экрана → Markdown + HTML)

```bash
# Базовый запуск
lecture-transcribe video.mp4 -o notes.md

# Только визуал (без аудио), с кэшированием
lecture-transcribe video.mp4 -o notes.md --no-audio \
  --save-keyframes ./keyframes \
  --save-visual ./visual.json \
  --cache

# Отладка
lecture-transcribe video.mp4 -o notes.md --debug
```

### Основные опции

| Опция | Описание |
|-------|----------|
| `-o, --output` | Путь к выходному .md файлу |
| `--subject` | Название предмета (для LLM-постобработки аудио) |
| `--terms` | Термины предмета через запятую (context bias для транскрипции) |
| `--cache` | Переиспользовать кэшированные промежуточные файлы |
| `--save-transcript PATH` | Сохранить транскрипт в JSON |
| `--postprocess-model` | Модель для постобработки (по умолчанию `mistral-large-latest`) |
| `--debug` | Подробное логирование |

**Только для видео-режима:**

| Опция | Описание |
|-------|----------|
| `--no-audio` | Пропустить аудио-пайплайн (только визуал) |
| `--save-keyframes DIR` | Сохранить ключевые кадры как PNG |
| `--save-visual PATH` | Сохранить результаты OCR в JSON (поддерживает resume) |
| `--fps` | Частота семплирования кадров (по умолчанию 2.0) |
| `--vision-model` | Модель для структурирования |
| `--vision-interval` | Минимальный интервал между API-вызовами в секундах |
| `--vision-rate-retries` | Количество повторов при 429 ошибке |
| `--continue-after-rate-limit` | Продолжать обработку после исчерпания rate limit |
| `--config` | Путь к YAML-файлу конфигурации |

## Выходные форматы

### Аудио → Obsidian Markdown

```markdown
---
subject: "Математический анализ"
source: "recording.aac"
date: 2026-02-18
type: lecture-notes
---

## Предел функции

> [!definition] Определение
> Число $L$ называется пределом функции $f(x)$ при $x \to a$, если
> $$\forall \varepsilon > 0 \; \exists \delta > 0 : 0 < |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon$$

> [!example] Пример
> Найти $\lim_{x \to 0} \frac{\sin x}{x}$.

> [!important] Важно
> Первый замечательный предел: $\lim_{x \to 0} \frac{\sin x}{x} = 1$
```

### Видео → Markdown + HTML

```markdown
# Конспект лекции

## [00:02:15]

**Лектор:** Рассмотрим функцию f(x) = 1/x на промежутке...

**На доске:**

Задана функция $f(x) = \frac{1}{x}$ на множестве $x \in [1; +\infty)$.
```

HTML генерируется автоматически с MathJax v3, адаптивным дизайном и dark mode.

## Работа с rate limit (бесплатный тариф)

На бесплатном research tier Mistral есть ограничения по количеству запросов. Пайплайн адаптирован для этого:

- **Инкрементальное сохранение** — каждый успешный результат OCR сразу записывается в `visual.json`
- **Resume** — при повторном запуске с `--cache` уже обработанные кадры пропускаются
- **Адаптивные интервалы** — при получении 429 интервал между запросами автоматически увеличивается
- **Graceful abort** — при исчерпании лимита обработка останавливается, сохраняя прогресс

## Структура проекта

```
lecture_transcriber/
├── cli.py                    # CLI (Click)
├── config.py                 # Конфигурация (dataclass + YAML + .env)
├── pipeline.py               # Оркестратор пайплайна
├── audio/
│   ├── extractor.py          # ffmpeg: audio/video → WAV
│   └── transcriber.py        # Voxtral API: audio → transcript
├── video/
│   ├── sampler.py            # OpenCV: video → frames (N fps)
│   ├── keyframe_detector.py  # SSIM + flicker filter
│   ├── keyframe_tracker.py   # Buffer matching (Phase 2 stub)
│   └── visual_recognizer.py  # OCR + LLM structuring → visual segments
├── llm/
│   └── postprocessor.py      # LLM постобработка транскрипта → Obsidian MD
├── merge/
│   ├── synchronizer.py       # Align audio + visual by timestamps
│   └── formatter.py          # Markdown + HTML generation
└── utils/
    └── api_client.py         # Mistral API wrapper (retry, rate limit, backoff)
```

## Конфигурация (YAML)

```yaml
# config.yaml (опционально)
mistral_api_key: "..."  # или через .env / MISTRAL_API_KEY

# Аудио-постобработка
subject: "Математический анализ"
postprocess_model: "mistral-large-latest"
postprocess_chunk_words: 8000

# Видео-пайплайн
sample_fps: 2.0
ssim_threshold_significant: 0.85
ssim_threshold_minor: 0.95
vision_model: "mistral-large-latest"
vision_min_interval: 8.0
vision_max_429_retries: 3
vision_abort_on_rate_limit: true

merge_window_seconds: 5.0
include_timestamps: true
include_audio_text: true
```

## Лицензия

MIT
