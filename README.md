# Lecture Scribe

CLI-инструмент на Python, который превращает видеозаписи лекций в структурированные конспекты (Markdown + HTML) с LaTeX-формулами и таймстемпами.

Разработан для обработки записей экрана из Telegram-звонков, где лектор пишет от руки в Paint.

> Проект использует **бесплатную исследовательскую лицензию Mistral API** (free research tier). Все API-вызовы — бесплатны в рамках этого тарифа.

## Как это работает

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

## Используемые модели Mistral

| Задача | Модель | Тип |
|--------|--------|-----|
| Транскрипция аудио | `voxtral-mini-latest` | Audio |
| OCR кадров | `mistral-ocr-latest` | OCR |
| Структурирование текста | `mistral-large-latest` | Chat |

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

```bash
# Базовый запуск
lecture-transcribe video.mp4 -o notes.md

# Только визуал (без аудио), с кэшированием промежуточных файлов
lecture-transcribe video.mp4 -o notes.md --no-audio \
  --save-keyframes ./keyframes \
  --save-transcript ./transcript.json \
  --save-visual ./visual.json \
  --cache

# С терминами предмета (context bias для транскрипции)
lecture-transcribe video.mp4 -o notes.md \
  --terms "определитель,матрица,собственное значение"

# Отладка
lecture-transcribe video.mp4 -o notes.md --debug
```

### Основные опции

| Опция | Описание |
|-------|----------|
| `-o, --output` | Путь к выходному .md файлу (HTML генерируется автоматически) |
| `--no-audio` | Пропустить аудио-пайплайн (только визуал) |
| `--cache` | Переиспользовать кэшированные промежуточные файлы |
| `--save-keyframes DIR` | Сохранить ключевые кадры как PNG |
| `--save-transcript PATH` | Сохранить транскрипт в JSON |
| `--save-visual PATH` | Сохранить результаты OCR в JSON (поддерживает resume) |
| `--terms` | Термины предмета через запятую (context bias) |
| `--fps` | Частота семплирования кадров (по умолчанию 2.0) |
| `--vision-model` | Модель для структурирования (по умолчанию `mistral-large-latest`) |
| `--vision-interval` | Минимальный интервал между API-вызовами в секундах |
| `--vision-rate-retries` | Количество повторов при 429 ошибке |
| `--continue-after-rate-limit` | Продолжать обработку после исчерпания rate limit |
| `--config` | Путь к YAML-файлу конфигурации |
| `--debug` | Подробное логирование |

## Работа с rate limit (бесплатный тарифф)

На бесплатном research tier Mistral есть ограничения по количеству запросов. Пайплайн адаптирован для этого:

- **Инкрементальное сохранение** — каждый успешный результат OCR сразу записывается в `visual.json`
- **Resume** — при повторном запуске с `--cache` уже обработанные кадры пропускаются
- **Адаптивные интервалы** — при получении 429 интервал между запросами автоматически увеличивается
- **Graceful abort** — при исчерпании лимита обработка останавливается, сохраняя прогресс

Типичный сценарий для длинной лекции:

```bash
# Запуск 1 — обработает первые N кадров до rate limit
lecture-transcribe video.mp4 -o notes.md --no-audio \
  --save-keyframes ./keyframes --save-visual ./visual.json --cache

# Подождать ~10-30 минут (восстановление квоты)

# Запуск 2 — продолжит с того места, где остановился
lecture-transcribe video.mp4 -o notes.md --no-audio \
  --save-keyframes ./keyframes --save-visual ./visual.json --cache
```

## Выходные форматы

### Markdown (`output.md`)

```markdown
# Конспект лекции

## [00:02:15]

**Лектор:** Рассмотрим функцию f(x) = 1/x на промежутке...

**На доске:**

Задана функция $f(x) = \frac{1}{x}$ на множестве $x \in [1; +\infty)$.

> **Схема:** График функции f(x) = 1/x, убывающей от 1 до 0.
```

### HTML (`output.html`)

Автоматически генерируется вместе с Markdown. Включает:
- MathJax v3 для рендеринга LaTeX-формул
- Адаптивный дизайн с dark mode
- Стилизованные блоки для таймстемпов, речи лектора и содержимого доски

## Структура проекта

```
lecture_transcriber/
├── cli.py                    # CLI (Click)
├── config.py                 # Конфигурация (dataclass + YAML + .env)
├── pipeline.py               # Оркестратор пайплайна
├── audio/
│   ├── extractor.py          # ffmpeg: video → WAV
│   └── transcriber.py        # Voxtral API: audio → transcript
├── video/
│   ├── sampler.py            # OpenCV: video → frames (N fps)
│   ├── keyframe_detector.py  # SSIM + flicker filter
│   ├── keyframe_tracker.py   # Buffer matching (Phase 2 stub)
│   └── visual_recognizer.py  # OCR + LLM structuring → visual segments
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
