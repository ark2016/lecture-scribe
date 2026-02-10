# Lecture Video Transcriber — Архитектура и ТЗ

## 1. Обзор проекта

### Что это
CLI-инструмент на Python, который принимает видеозапись лекции (запись экрана Paint/рисование от руки + голос лектора через ТГ-звонок) и генерирует структурированные конспекты в Markdown с LaTeX-формулами, привязанные к таймлайну.

### Ключевые особенности входных данных
- Лектор пишет от руки в Paint (или аналогичном редакторе) сверху вниз
- Контент: рукописный текст на русском, математические формулы, схемы/диаграммы
- Лектор **иногда возвращается** на полпараграфа назад и правит/дописывает
- Лектор **иногда правит** текст в текущем месте (зачёркивание, перезапись)
- Видео **мигает** из-за артефактов ТГ-звонка (кратковременные визуальные глитчи)
- Аудио — голос лектора с комментариями к тому, что он пишет на экране

### Стек
- **Python 3.11+**
- **Mistral API** — Voxtral Mini Transcribe V2 (аудио), Mistral Vision (кадры)
- **OpenCV** — извлечение и сравнение кадров
- **ffmpeg** — извлечение аудио из видео
- Выход: **Markdown** с LaTeX-блоками и таймстемпами

---

## 2. Архитектура

### Общая схема пайплайна

```
video.mp4
│
├──[Stage 1: Audio Pipeline]──────────────────────────────
│   ffmpeg extract audio
│   └─→ audio.wav
│       └─→ Voxtral Transcribe (word-level timestamps)
│           └─→ transcript.json
│               [{text: "...", start: 12.3, end: 12.8}, ...]
│
├──[Stage 2: Visual Pipeline]─────────────────────────────
│   OpenCV frame sampling (1-2 fps)
│   └─→ raw frames
│       └─→ Keyframe Detector (SSIM + flicker filter)
│           └─→ keyframes[]
│               └─→ Keyframe Tracker (buffer matching)
│                   └─→ events[]
│                       [{type: "new"|"update"|"return", 
│                         frame, timestamp, matched_kf_id}, ...]
│                       └─→ Mistral Vision (batch)
│                           └─→ visual_segments.json
│                               [{id, timestamp, text, latex, 
│                                 diagram_desc, raw_response}, ...]
│
└──[Stage 3: Merge & Output]──────────────────────────────
    transcript.json + visual_segments.json
    └─→ Synchronizer (align by timestamps)
        └─→ lecture_notes.md
```

### Модули

```
lecture_transcriber/
├── __init__.py
├── cli.py                  # CLI entry point (argparse/click)
├── config.py               # Настройки, API ключи, пороги
├── pipeline.py             # Оркестратор всего пайплайна
│
├── audio/
│   ├── __init__.py
│   ├── extractor.py        # ffmpeg: video → audio
│   └── transcriber.py      # Voxtral API: audio → transcript
│
├── video/
│   ├── __init__.py
│   ├── sampler.py          # OpenCV: video → raw frames (1-2 fps)
│   ├── keyframe_detector.py    # SSIM, flicker filter, change detection
│   ├── keyframe_tracker.py     # Buffer matching, event classification
│   └── visual_recognizer.py    # Mistral Vision API: frames → text/latex
│
├── merge/
│   ├── __init__.py
│   ├── synchronizer.py     # Align audio + visual by timestamps
│   └── formatter.py        # Generate final Markdown
│
└── utils/
    ├── __init__.py
    ├── image.py            # SSIM, image diff, region comparison
    └── api_client.py       # Mistral API wrapper with retry/rate-limit
```

---

## 3. Детальное описание модулей

### 3.1 Audio Extractor (`audio/extractor.py`)

**Вход:** путь к видеофайлу
**Выход:** путь к WAV-файлу

```python
def extract_audio(video_path: str, output_path: str) -> str:
    """ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav"""
```

- Моно, 16kHz — оптимально для ASR
- Если файл >100MB, разбить на чанки (Voxtral поддерживает до 3 часов, но для надёжности)

### 3.2 Audio Transcriber (`audio/transcriber.py`)

**Вход:** путь к WAV
**Выход:** список сегментов с word-level timestamps

```python
@dataclass
class TranscriptWord:
    text: str
    start: float  # секунды
    end: float

@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: list[TranscriptWord]

def transcribe(audio_path: str, language: str = "ru", 
               context_bias: list[str] | None = None) -> list[TranscriptSegment]:
    """
    Вызов Voxtral Mini Transcribe V2 API.
    
    - model: "voxtral-mini-latest"
    - endpoint: /v1/audio/transcriptions
    - timestamp_granularities: ["word", "segment"]
    - language: "ru"
    - context_bias: список терминов предмета (до 100 слов)
    """
```

**Важно:**
- `context_bias` — передать термины из предмета (для context biasing, экспериментально для русского, но может помочь)
- Результат сохранять в JSON для кэширования

### 3.3 Frame Sampler (`video/sampler.py`)

**Вход:** путь к видео
**Выход:** генератор кортежей (timestamp, frame)

```python
def sample_frames(video_path: str, fps: float = 2.0) -> Iterator[tuple[float, np.ndarray]]:
    """
    Извлекает кадры с заданной частотой.
    
    fps=2.0 — каждые 0.5 секунды. Для Paint-лекций достаточно,
    лектор не пишет быстрее.
    """
```

- 2 FPS достаточно для отслеживания рукописного текста
- Можно уменьшить до 1 FPS для экономии

### 3.4 Keyframe Detector (`video/keyframe_detector.py`)

**Вход:** поток кадров из sampler
**Выход:** список ключевых кадров с метаданными

```python
@dataclass
class KeyframeEvent:
    timestamp: float
    frame: np.ndarray
    change_type: str  # "significant" | "minor" | "flicker"
    change_region: tuple[int, int, int, int] | None  # bbox области изменения
    ssim_vs_prev: float

def detect_keyframes(
    frames: Iterator[tuple[float, np.ndarray]],
    ssim_threshold_significant: float = 0.85,  # ниже — значительное изменение
    ssim_threshold_minor: float = 0.95,         # между minor и significant — мелкая правка
    flicker_window: int = 3,                     # кадров для детекции мерцания
    flicker_recovery_threshold: float = 0.95,    # порог "вернулся обратно"
) -> Iterator[KeyframeEvent]:
```

**Алгоритм фильтрации мерцания (ТГ-артефакты):**

```
Для каждого кадра frame[i]:
  1. ssim(frame[i], frame[i-1]) < threshold → кандидат на изменение
  2. Проверить: ssim(frame[i+1], frame[i-1]) > flicker_recovery_threshold?
     - Да → это мерцание, ПРОПУСТИТЬ frame[i]
     - Нет → реальное изменение
  3. Проверить: ssim(frame[i+2], frame[i-1]) > flicker_recovery_threshold?
     - Да → мерцание длиной в 2 кадра, ПРОПУСТИТЬ
     - Нет → подтверждённое изменение, EMIT
```

**Определение области изменения:**

```python
def compute_change_region(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[int, int, int, int]:
    """
    Абсолютная разность → порог → bounding box ненулевой области.
    Возвращает (y_min, y_max, x_min, x_max) — где произошло изменение.
    
    Это нужно для определения: лектор пишет внизу (продолжение)
    или правит что-то вверху (возврат).
    """
```

### 3.5 Keyframe Tracker (`video/keyframe_tracker.py`)

**Ключевой модуль.** Отслеживает «состояние доски» и определяет, куда лектор вернулся.

```python
@dataclass
class TrackedKeyframe:
    id: str                    # уникальный ID
    timestamp_first: float     # когда впервые появился
    timestamp_last: float      # когда последний раз обновлён
    frame: np.ndarray          # текущее состояние
    history: list[tuple[float, np.ndarray]]  # история версий
    
@dataclass  
class VisualEvent:
    timestamp: float
    event_type: str            # "new_content" | "update_existing" | "return_to_previous"
    keyframe_id: str           # к какому TrackedKeyframe относится
    frame: np.ndarray          # текущий кадр
    previous_frame: np.ndarray | None  # предыдущее состояние (для diff)

class KeyframeTracker:
    def __init__(
        self,
        buffer_size: int = 50,           # макс кол-во ключевых кадров в буфере
        match_threshold: float = 0.80,    # порог для "это тот же кадр"
        update_threshold: float = 0.90,   # порог для "мелкая правка существующего"
    ):
        self.buffer: list[TrackedKeyframe] = []
    
    def process(self, keyframe_event: KeyframeEvent) -> VisualEvent:
        """
        Алгоритм:
        
        1. Сравнить текущий кадр со ВСЕМИ кадрами в буфере
           (SSIM, или feature embeddings для устойчивости)
        
        2. Найти лучшее совпадение (best_match, best_score)
        
        3. Классификация:
           a) best_score > update_threshold (>0.90):
              → "update_existing": мелкая правка в текущем месте
              → обновить frame в best_match
              
           b) match_threshold < best_score <= update_threshold (0.80-0.90):
              → "return_to_previous": лектор вернулся к старому месту
                 и внёс заметные изменения
              → сохранить предыдущую версию в history
              → обновить frame в best_match
              
           c) best_score <= match_threshold (<0.80):
              → "new_content": совершенно новый контент
              → создать новый TrackedKeyframe, добавить в буфер
        
        4. Для "return_to_previous":
           Дополнительная проверка — сравнить change_region:
           - Если изменение в верхней части кадра → подтверждает возврат
           - Если изменение в нижней части → может быть продолжение,
             а не возврат (уточнить через VLM)
        """
```

**Оптимизация сравнения с буфером:**
- Не нужно сравнивать со всеми 50 кадрами через SSIM (дорого)
- Можно использовать **perceptual hash** (pHash) для быстрого предварительного отсева
- Полный SSIM только для кандидатов с близким хешем
- Или: уменьшить кадры до 64x64 и сравнивать гистограммы цветов

### 3.6 Visual Recognizer (`video/visual_recognizer.py`)

**Вход:** список VisualEvent
**Выход:** список визуальных сегментов с распознанным текстом

```python
@dataclass
class VisualSegment:
    keyframe_id: str
    timestamp: float
    event_type: str
    text: str              # распознанный рукописный текст
    latex: str             # формулы в LaTeX
    diagram_description: str  # описание схем/диаграмм
    raw_response: str      # полный ответ VLM для отладки

def recognize_visual_content(events: list[VisualEvent]) -> list[VisualSegment]:
    """
    Для каждого VisualEvent вызывает Mistral Vision API.
    
    Стратегия промптов зависит от event_type:
    
    1. "new_content":
       Отправить: текущий кадр
       Промпт: PROMPT_NEW_CONTENT
    
    2. "update_existing":
       Отправить: предыдущий кадр + текущий кадр
       Промпт: PROMPT_UPDATE
    
    3. "return_to_previous":
       Отправить: предыдущий кадр + текущий кадр
       Промпт: PROMPT_RETURN
    """
```

**Промпты:**

```python
PROMPT_NEW_CONTENT = """
Ты анализируешь кадр из видеолекции. На экране — рукописный текст и/или формулы,
написанные в графическом редакторе (Paint).

Задача:
1. Запиши весь видимый рукописный текст (русский язык)
2. Все математические формулы запиши в LaTeX (обёрнуты в $..$ или $$..$$)
3. Если есть схемы или диаграммы — опиши их структуру и содержание
4. Сохраняй порядок сверху вниз, слева направо

Формат ответа — JSON:
{
    "text": "распознанный текст с формулами в LaTeX",
    "latex_blocks": ["\\\\frac{a}{b}", ...],
    "diagrams": ["описание схемы 1", ...]
}
"""

PROMPT_UPDATE = """
Ты анализируешь два кадра из видеолекции: предыдущее и текущее состояние экрана.
Лектор внёс изменения (правка, дописывание).

Задача:
1. Определи, что именно изменилось между двумя кадрами
2. Запиши ТОЛЬКО новый или изменённый контент
3. Формулы — в LaTeX

Формат ответа — JSON:
{
    "change_description": "что изменилось",
    "new_text": "новый/изменённый текст",
    "latex_blocks": [...],
    "diagrams": [...]
}
"""

PROMPT_RETURN = """
Ты анализируешь два кадра из видеолекции. Лектор ВЕРНУЛСЯ к ранее написанному
фрагменту и внёс изменения (правка, дополнение).

Задача:
1. Определи, к какому месту вернулся лектор
2. Запиши, что именно было изменено или добавлено
3. Формулы — в LaTeX

Формат ответа — JSON:
{
    "return_context": "к какому фрагменту вернулся",
    "change_description": "что изменилось",
    "updated_text": "обновлённый текст фрагмента",
    "latex_blocks": [...],
    "diagrams": [...]
}
"""
```

**Оптимизация API-вызовов:**
- Батчить запросы (Mistral Vision поддерживает несколько картинок в одном запросе)
- Кэшировать результаты — если кадр уже распознан и не изменился, не вызывать повторно
- Для "update_existing" с мелкой правкой можно пропускать VLM и ограничиться diff

**Выбор модели:**
- `mistral-small-latest` — дешевле, быстрее, достаточно для чистого текста
- `mistral-large-latest` — для сложных формул и схем
- Рекомендация: начать с `mistral-small-latest`, переключить на large если качество недостаточно

### 3.7 Synchronizer (`merge/synchronizer.py`)

**Вход:** transcript segments + visual segments
**Выход:** синхронизированный таймлайн

```python
@dataclass
class LectureBlock:
    timestamp_start: float
    timestamp_end: float
    audio_text: str              # что лектор говорил
    visual_content: str          # что было на экране
    latex_blocks: list[str]
    diagram_descriptions: list[str]
    block_type: str              # "content" | "revision" | "return"

def synchronize(
    transcript: list[TranscriptSegment],
    visual_segments: list[VisualSegment],
    merge_window: float = 5.0,  # секунд — окно для привязки аудио к кадру
) -> list[LectureBlock]:
    """
    Алгоритм синхронизации:
    
    1. Создать единый таймлайн из обоих источников
    2. Для каждого visual_segment найти audio_segments,
       попадающие в окно [visual.timestamp - merge_window, visual.timestamp + merge_window]
    3. Объединить в LectureBlock
    4. Аудио-сегменты, не привязанные к визуалу — 
       отдельные блоки "commentary"
    5. Для "return" событий — пометить, что этот блок
       обновляет более ранний блок (ссылка по keyframe_id)
    """
```

### 3.8 Formatter (`merge/formatter.py`)

**Вход:** список LectureBlock
**Выход:** Markdown-файл

```python
def format_markdown(
    blocks: list[LectureBlock],
    title: str = "Конспект лекции",
    include_timestamps: bool = True,
    include_audio: bool = True,
) -> str:
```

**Формат выходного Markdown:**

```markdown
# Конспект лекции — [Название]

## [00:00:00] Введение

**Лектор:** Сегодня мы рассмотрим свойства определителей матриц...

**На доске:**

Определитель матрицы $A$ обозначается $\det(A)$ или $|A|$.

Свойства:
1. $\det(AB) = \det(A) \cdot \det(B)$
2. $\det(A^T) = \det(A)$

## [00:05:23] Пример вычисления

**Лектор:** Рассмотрим матрицу три на три...

**На доске:**

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

> **Схема:** Диаграмма правила Саррюса — три диагонали слева направо
> со знаком плюс, три диагонали справа налево со знаком минус.

---
*[00:08:12] ← Возврат к [00:05:23]: лектор исправил элемент $a_{33}$ с $9$ на $10$*

**Исправление на доске:**

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{pmatrix}$$
```

### 3.9 API Client (`utils/api_client.py`)

```python
class MistralClient:
    """
    Обёртка над Mistral API с:
    - Retry с exponential backoff
    - Rate limiting (учёт RPS квоты)
    - Кэширование результатов
    - Логирование вызовов и стоимости
    """
    
    def __init__(self, api_key: str, max_retries: int = 3, rps_limit: float = 1.0):
        ...
    
    def transcribe_audio(self, audio_path: str, **kwargs) -> dict:
        """Voxtral Transcribe API"""
        
    def analyze_image(self, images: list[str], prompt: str, 
                      model: str = "mistral-small-latest") -> dict:
        """Vision API — принимает base64-картинки"""
```

**Rate limiting:**
- У тебя 1 RPS квота → клиент должен выдерживать минимум 1 секунду между запросами
- Для визуальной части это значит: ~60 VLM-вызовов в минуту максимум
- Для часовой лекции с ~30-50 ключевыми кадрами — укладываемся за 1-2 минуты

---

## 4. Конфигурация

```python
# config.py

@dataclass
class Config:
    # API
    mistral_api_key: str          # env: MISTRAL_API_KEY
    rps_limit: float = 1.0        # запросов в секунду
    
    # Audio
    audio_sample_rate: int = 16000
    transcription_language: str = "ru"
    context_bias_terms: list[str] = field(default_factory=list)
    
    # Video sampling
    sample_fps: float = 2.0       # кадров в секунду
    
    # Keyframe detection
    ssim_threshold_significant: float = 0.85
    ssim_threshold_minor: float = 0.95
    flicker_window: int = 3
    flicker_recovery_threshold: float = 0.95
    
    # Keyframe tracking
    buffer_size: int = 50
    match_threshold: float = 0.80
    update_threshold: float = 0.90
    use_phash_prefilter: bool = True
    
    # Visual recognition
    vision_model: str = "mistral-small-latest"
    skip_minor_updates: bool = True  # не вызывать VLM для мелких правок
    
    # Merge
    merge_window_seconds: float = 5.0
    
    # Output
    include_timestamps: bool = True
    include_audio_text: bool = True
    output_format: str = "markdown"  # "markdown" | "html"
```

Конфиг загружается из YAML-файла или CLI-аргументов.

---

## 5. CLI интерфейс

```bash
# Базовое использование
lecture-transcribe video.mp4 -o notes.md

# С терминами предмета (context bias)
lecture-transcribe video.mp4 -o notes.md \
  --terms "определитель,матрица,собственное значение,Крамер"

# С настройками
lecture-transcribe video.mp4 -o notes.md \
  --config config.yaml \
  --vision-model mistral-large-latest \
  --fps 1.0 \
  --no-audio  # только визуал

# Отладка — сохранить промежуточные результаты
lecture-transcribe video.mp4 -o notes.md \
  --debug \
  --save-keyframes ./keyframes/ \
  --save-transcript ./transcript.json \
  --save-visual ./visual.json
```

---

## 6. Зависимости

```toml
# pyproject.toml
[project]
name = "lecture-transcriber"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "mistralai",           # Mistral Python SDK
    "opencv-python",       # cv2 — кадры, SSIM, diff
    "numpy",               # массивы
    "scikit-image",        # SSIM (skimage.metrics)
    "imagehash",           # perceptual hashing
    "click",               # CLI
    "pyyaml",              # конфиг
    "rich",                # progress bars, logging
    "tenacity",            # retry logic
]

[project.scripts]
lecture-transcribe = "lecture_transcriber.cli:main"
```

Внешние: `ffmpeg` — должен быть установлен в системе.

---

## 7. Порядок реализации (рекомендуемый)

### Фаза 1 — MVP (работающий пайплайн без трекинга)
1. `audio/extractor.py` — ffmpeg wrapper
2. `audio/transcriber.py` — Voxtral API
3. `video/sampler.py` — извлечение кадров
4. `video/keyframe_detector.py` — базовая детекция (SSIM с предыдущим кадром + фильтр мерцания)
5. `video/visual_recognizer.py` — Mistral Vision для каждого ключевого кадра (только PROMPT_NEW_CONTENT)
6. `merge/synchronizer.py` — наивная синхронизация по ближайшему таймстемпу
7. `merge/formatter.py` — генерация Markdown
8. `cli.py` + `pipeline.py` — собрать в CLI

**Результат:** работающий инструмент, который обрабатывает простые лекции без возвратов.

### Фаза 2 — Трекинг и возвраты
9. `keyframe_tracker.py` — буфер ключевых кадров, матчинг, классификация событий
10. Обновить `visual_recognizer.py` — разные промпты для разных типов событий (PROMPT_UPDATE, PROMPT_RETURN)
11. Обновить `synchronizer.py` — обработка "return_to_previous" событий
12. Обновить `formatter.py` — отображение возвратов и правок в Markdown

### Фаза 3 — Оптимизация и polish
13. `utils/api_client.py` — полноценный rate limiter, кэширование, логирование стоимости
14. pHash для быстрого pre-filtering в tracker
15. Батчинг VLM-запросов
16. Конфиг из YAML
17. `--debug` режим с сохранением промежуточных данных
18. Тесты

---

## 8. Потенциальные проблемы и решения

| Проблема | Решение |
|----------|---------|
| SSIM нестабилен при мелких изменениях почерка | Использовать pHash или feature embeddings для буферного матчинга, SSIM только для соседних кадров |
| VLM плохо читает мелкий почерк | Кропнуть только изменившуюся область и отправить в увеличенном виде |
| Мерцание ТГ-звонка разной длительности | Настраиваемый flicker_window (3-5 кадров), проверка восстановления с несколькими предыдущими |
| Rate limit 1 RPS | Приоритизировать VLM-вызовы: пропускать minor updates, батчить где возможно |
| Формулы распознаются неточно | Отправлять в VLM увеличенный кроп области с формулой + контекст из аудио ("лектор говорит: интеграл от...") |
| Аудио и визуал рассинхронизированы | Расширить merge_window, использовать содержательное сопоставление (упоминание термина в аудио ↔ появление на экране) |

---

## 9. Пример запуска и ожидаемый результат

**Вход:**
```
lecture_calculus_01.mp4  (1 час, 720p, ТГ-запись)
```

**Промежуточные данные:**
- `transcript.json` — ~3000-5000 слов транскрипта с timestamps
- `keyframes/` — ~30-80 ключевых кадров
- `visual.json` — распознанный контент для каждого кадра

**Выход:**
- `notes.md` — структурированный конспект, ~5-15 страниц
- Время обработки: ~5-10 минут (зависит от API latency)
- Стоимость API: ~$0.20-0.50 за лекцию (транскрипция + VLM), на данный момент у меня исследовательская лицензия, т.е. всё бесплатно