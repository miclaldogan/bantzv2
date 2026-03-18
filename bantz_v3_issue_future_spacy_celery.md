# Issue #10 — [future] Upgrade memory NER: LLM-based → spaCy pipeline

**Labels:** `enhancement`, `layer:memory`, `future`, `good first issue`

> ⏸ Bu issue şimdi açılır ama **Neo4j memory layer (Issue #6) stable olduktan sonra** ele alınır.

---

## Summary

Issue #6'da Neo4j'e entity yazmak için LLM prompt tabanlı NER kullanılıyor ("bu cümleden entity çıkar, JSON döndür"). Bu çalışır ama her memory write için bir LLM çağrısı demek — yavaş ve token harcar. spaCy ile bu adım local, hızlı ve ücretsiz hale gelir.

---

## Motivation

Şu an planlanan LLM-NER akışı:

```
konuşma turu
    → Gemini/Qwen'e "entity çıkar" prompt'u   ← her write'ta token harcar
    → JSON parse
    → Neo4j write
```

spaCy ile hedef akış:

```
konuşma turu
    → spaCy NER (local, ~5ms)                 ← sıfır token, sıfır API çağrısı
    → entity + relation extraction
    → Neo4j write
```

---

## Neden Şimdi Değil?

- spaCy paketi **32 MB** + dil modeli **~12 MB** — küçük değil
- LLM-based NER Issue #6 için yeterince iyi bir başlangıç
- spaCy entegrasyonu Neo4j şeması stable olmadan yazılırsa iki kez refactor olur
- **Önce çalışır hale getir, sonra optimize et**

---

## Acceptance Criteria (zamanı gelince)

- [ ] `pip install spacy && python -m spacy download en_core_web_sm` kurulum script'e eklenir
- [ ] `bantz/memory/ner.py` → `NERExtractor` class, iki backend destekler: `"spacy"` ve `"llm"`
- [ ] Backend `config.yaml`'dan seçilebilir: `memory.ner_backend: spacy | llm`
- [ ] `NERExtractor.extract(text)` → `list[Entity(text, label, start, end)]` döner
- [ ] Desteklenen entity tipleri: `PERSON`, `ORG`, `DATE`, `TIME`, `GPE`, `EVENT`, `PRODUCT`
- [ ] Türkçe metin için `xx_ent_wiki_sm` (multilingual) modeli değerlendirilir
- [ ] LLM backend korunur — Türkçe edge case'lerde spaCy yetersiz kalabilir, fallback olarak kalır
- [ ] Benchmark: spaCy vs LLM NER, 100 örnek cümle üzerinde F1 karşılaştırması

---

## Implementation Sketch

```python
# bantz/memory/ner.py

class NERExtractor:
    def __init__(self, backend: str = "llm"):
        self.backend = backend
        if backend == "spacy":
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> list[Entity]:
        if self.backend == "spacy":
            return self._spacy_extract(text)
        return self._llm_extract(text)

    def _spacy_extract(self, text: str) -> list[Entity]:
        doc = self.nlp(text)
        return [
            Entity(ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in doc.ents
        ]

    async def _llm_extract(self, text: str) -> list[Entity]:
        # mevcut LLM prompt yöntemi — fallback olarak kalır
        prompt = f"Extract named entities from: '{text}'. Return JSON array."
        ...
```

---

## Trigger Condition

Bu issue'yu ele almak için şu kriterlerin sağlanmış olması gerekir:

1. Issue #6 (Neo4j memory layer) production'da stable
2. Günlük memory write sayısı > 50 olduğunda LLM NER maliyeti hissedilir hale gelir
3. Token tasarrufu öncelik haline gelirse

---
---

# Issue #11 — [future] Replace APScheduler with Celery for parallel task execution

**Labels:** `enhancement`, `layer:scheduler`, `future`

> ⏸ Bu issue şimdi açılır ama **APScheduler (Issue #8) yetersiz kaldığında** ele alınır.

---

## Summary

Issue #8'de APScheduler ile cron + one-shot görevler yönetiliyor. Tek bir görev akışı için bu yeterli. Eğer Bantz aynı anda birden fazla ağır görevi paralel işlemesi gerekirse (örn. gece 5 farklı script, paralel web scraping, uzun süreli kod çalıştırma) Celery worker mimarisine geçilir.

---

## Motivation

APScheduler sınırı: tek process içinde çalışır, bir görev bloklanırsa diğerleri bekler. Celery sınırı yok — N worker process, distributed queue, retry mekanizması, görev izleme (Flower UI).

**Senaryo örneği (Celery gerektirir):**
```
gece 02:00 →  task_1: TEKNOFEST verisi çek (30 dk)
              task_2: kod test suite çalıştır (15 dk)
              task_3: haftalık özet raporu üret (10 dk)
              → hepsi paralel, bağımsız worker'larda
```

**Şu anki senaryo (APScheduler yeterli):**
```
gece 02:00 → tek görev: "şu scripti çalıştır"
             sabah hazır olsun
```

---

## Neden Şimdi Değil?

- Celery = ayrı worker daemon + broker (Redis zaten var, o tamam) + monitoring
- Bantz şu an tek kullanıcı, tek makine → paralel worker overkill
- APScheduler Redis job store ile restart'tan kurtulur, bu yeterli
- **Celery'nin getirisi ancak eşzamanlı 3+ ağır görev olduğunda pozitife geçer**

---

## Acceptance Criteria (zamanı gelince)

- [ ] `pip install celery[redis] flower` kurulum script'e eklenir
- [ ] `bantz/scheduler/celery_app.py` → Celery app, Redis broker ile konfig
- [ ] Mevcut APScheduler job'ları `@celery_app.task` decorator'a migrate edilir
- [ ] `bantz/scheduler/worker.py` → `celery -A bantz.scheduler worker --loglevel=info`
- [ ] Flower monitoring UI: `localhost:5555` üzerinden görev izleme
- [ ] APScheduler tamamen kaldırılmaz — lightweight cron için `hybrid` mod: basit job'lar APScheduler, ağır paralel job'lar Celery
- [ ] systemd service dosyası: `bantz-worker.service` (boot'ta otomatik başlar)
- [ ] Retry policy: başarısız görevler 3x retry, exponential backoff

---

## Implementation Sketch

```python
# bantz/scheduler/celery_app.py
from celery import Celery

app = Celery(
    "bantz",
    broker="redis://localhost:6379/1",
    backend="redis://localhost:6379/2",
)

app.conf.update(
    task_serializer="json",
    result_expires=86400,  # 24h
    worker_max_tasks_per_child=100,
)

# Görev tanımı
@app.task(bind=True, max_retries=3)
def run_nightly_script(self, script_path: str):
    try:
        result = SystemTool().run(f"python {script_path}")
        return result.stdout
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

```bash
# systemd service
[Unit]
Description=Bantz Celery Worker
After=network.target redis.service

[Service]
WorkingDirectory=/home/iclaldogan/bantzv3
ExecStart=.venv/bin/celery -A bantz.scheduler worker --loglevel=info
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Trigger Condition

Bu issue'yu ele almak için şu kriterlerin sağlanmış olması gerekir:

1. Issue #8 (APScheduler) production'da stable
2. Aynı anda 3+ paralel ağır görev ihtiyacı doğar
3. Bir görevin bloklaması diğerlerini geciktirdiği gözlemlenir

---

## Related

- Issue #7 (Redis — Celery broker olarak zaten kurulu)
- Issue #8 (APScheduler — hybrid modda korunur)
- Issue #4 (SystemTool — Celery task'ları bunu çağırır)

---

*Generated for: bantz-v3 · March 2026*
