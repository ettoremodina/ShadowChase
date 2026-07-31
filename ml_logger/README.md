# ML Logger

`ml_logger` è un logger locale e riutilizzabile per esperimenti ML. Unifica
logging Python, metriche, parametri, artifact, telemetria, dashboard terminale
e report HTML sotto il lifecycle di una singola run.

Per una panoramica visuale e navigabile, aprire
[`docs/ml_logger_guide.html`](docs/ml_logger_guide.html) direttamente in un
browser.

Il core non contiene concetti del progetto che lo utilizza. In questa
repository, la registrazione delle partite è implementata separatamente in
`integrations/regicide_logging.py`.

## Installazione

Durante lo sviluppo:

```bash
python -m pip install -e .
```

Il nome della distribuzione è `modin-ml-logger`; il package importabile resta
`ml_logger`.

Dipendenze runtime:

- `rich` per console e dashboard live;
- `PyYAML` per la configurazione;
- `psutil` per la telemetria di processo e di sistema;
- SQLite, incluso nella standard library.

## Quickstart

```python
from ml_logger import get_logger, run

logger = get_logger(__name__)

with run(
    "training",
    name="resnet-baseline",
    config={"learning_rate": 0.001, "batch_size": 64},
) as experiment:
    experiment.log_params({"optimizer": "adam"})

    for epoch in range(10):
        loss = train_one_epoch()
        accuracy = evaluate()

        experiment.log_metrics(
            epoch,
            {
                "train/loss": loss,
                "eval/accuracy": accuracy,
            },
        )
        experiment.log_progress(epoch + 1, 10, "Training")

    model_path = save_model()
    experiment.log_artifact(model_path, kind="model")
    logger.info("Training completed")
```

Il context manager:

- marca la run `completed` quando il blocco termina;
- marca la run `failed` e salva l'errore se un'eccezione esce dal blocco;
- ferma telemetria e dashboard;
- chiude tutti gli handler installati dal package;
- genera gli export JSONL e il report configurato.

Non è necessario chiamare manualmente `complete()`, `fail()` o `stop()`.

## API pubblica

### `run(...)` e `run_scope(...)`

Sono alias e rappresentano l'API raccomandata. Accettano:

- `run_type`: categoria stabile del workflow, usata anche per gli override;
- `name`: nome leggibile della singola esecuzione;
- `config`: configurazione effettiva dell'esperimento;
- `root_dir`: destinazione degli artifact;
- `metadata`: provenienza aggiuntiva JSON-compatible;
- `logger_config_path`: configurazione esplicita del logger.

### `get_logger(name)`

Restituisce un normale `logging.Logger`. Il codice applicativo può continuare a
usare `logger.info`, `warning`, `error` e `exception`. Il logger non sostituisce
il modulo `logging`: ne configura soltanto gli handler posseduti.

### Metodi di `RunContext`

```python
experiment.log_params({"seed": 42})
experiment.log_summary({"best_epoch": 8})
experiment.log_metric("train/loss", 0.25, step=3)
experiment.log_metrics(3, {"train/loss": 0.25, "eval/f1": 0.91})
experiment.log_progress(completed=4, total=10, description="Epochs")
experiment.log_artifact("model.pt", kind="model")
experiment.save_result("summary.json", {"best_f1": 0.91})
```

`log_metrics` è preferibile a molte chiamate `log_metric`: una singola
transazione contiene tutte le misure relative allo stesso step.

### Compatibilità

`start_run()` e `RunLogger` restano disponibili per i workflow esistenti.
Richiedono però finalizzazione manuale:

```python
context = start_run("legacy")
try:
    ...
    context.complete()
except Exception as error:
    context.fail(error)
    raise
```

`DashboardLogger` è mantenuto soltanto come facciata di migrazione. Nuovo
codice non deve istanziarlo: la vista live viene attivata automaticamente da
`run()` o `start_run()`.

## Configurazione

Ordine di selezione:

1. file passato con `logger_config_path`;
2. variabile d'ambiente `ML_LOGGER_CONFIG`;
3. `logger_config.yaml` nella directory corrente;
4. configurazione predefinita inclusa nel package.

Il file scelto viene fuso ricorsivamente sopra i default. Gli override in
`run_type_overrides.<run_type>` vengono applicati per ultimi.

La configurazione viene letta una volta, all'apertura della run. Modificare
`logger_config.yaml` mentre un processo è già attivo non riconfigura quella
run: la modifica ha effetto dal comando successivo.

Esempio completo:

```yaml
version: 2

artifacts:
  root_dir: artifacts
  directories: [logs, metrics, artifacts, models]

logging:
  enabled: true
  level: INFO
  console: true
  file: true

saving:
  enabled: true
  params: true
  metrics: true
  results: true
  artifacts: true
  telemetry: true

metrics:
  include: ["train/*", "eval/*"]
  exclude: ["debug/*"]

dashboard:
  mode: auto
  refresh_rate: 4
  max_log_lines: 50
  max_metrics: 16
  screen: true
  metrics:
    include: ["train/loss", "eval/*"]
    exclude: []

telemetry:
  enabled: true
  sample_interval_sec: 5
  include_process: true
  include_system: true
  gpu_provider: auto

report:
  enabled: true
  filename: run_report.html
  visualization: auto
  max_charts: 20
  metrics:
    include: ["*"]
    exclude: ["debug/*"]

run_type_overrides:
  benchmark:
    telemetry:
      enabled: false
    dashboard:
      mode: compact
```

### Selezionare cosa salvare

`saving.enabled` è l'interruttore generale. Gli altri campi controllano le
singole categorie.

`metrics.include` e `metrics.exclude` accettano pattern compatibili con
`fnmatch`, per esempio `train/*`, `eval/accuracy` o `*loss`.

### Selezionare la visualizzazione

`dashboard.mode` accetta:

- `auto`: vista live solo in un terminale interattivo e nel processo padre;
- `live`: dashboard Rich a schermo intero;
- `compact`: normali righe di log Rich;
- `off`: nessuna dashboard; `logging.console` continua a controllare la console.

`dashboard.metrics` stabilisce quali metriche persistite vengono mostrate.
Nascondere una metrica nella dashboard non ne impedisce il salvataggio.
Il pannello `Log Stream` conserva fino a `max_log_lines` messaggi e mostra
sempre le ultime righe che entrano nello spazio disponibile del terminale.
Una metrica appare nella dashboard non appena il codice chiama `log_metrics`:
per fasi lunghe conviene pubblicare valori intermedi, non soltanto il riepilogo
finale.

`report.visualization` accetta:

- `auto`: tabella riepilogativa e grafici per le serie con più punti;
- `line`: grafici lineari e riepilogo;
- `table`: solo riepilogo.

Il report è un singolo HTML portabile, senza server e senza asset esterni.

## Telemetria

Il sampler gira in background ed è indipendente dal refresh della dashboard.
Non viene quindi eseguito `nvidia-smi` dentro il ciclo di rendering.

Campi disponibili:

- sistema: CPU, RAM usata/totale e disco;
- processo: PID, CPU, RSS e numero di thread;
- NVIDIA GPU: utilizzo, memoria, temperatura, indice e nome.

Se il provider GPU non è disponibile, viene registrato
`available: false`; l'assenza del provider non viene confusa con utilizzo zero.

`sample_interval_sec` controlla il campionamento. La persistenza richiede sia
`telemetry.enabled` sia `saving.telemetry`.

## Storage e formati

```text
artifacts/
├── catalog.sqlite
└── runs/
    └── YYYY-MM-DD/
        └── <run-id>/
            ├── manifest.json
            ├── config.json
            ├── logs/run.log
            ├── metrics/metrics.jsonl
            ├── metrics/telemetry.jsonl
            ├── artifacts/
            └── reports/run_report.html
```

SQLite è la fonte canonica per run ed eventi. `metrics.jsonl` e
`telemetry.jsonl` sono viste di compatibilità ricostruite alla fine della run.
Il manifest usa `schema_version: 2`.

Gli eventi persistiti comprendono:

- `run.started`, `run.completed`, `run.failed`;
- `params`, `metrics`, `telemetry`;
- `artifact`, `result`.

I log testuali restano in `logs/run.log`; non vengono duplicati nel database.

## Multiprocessing

Il processo padre possiede dashboard e file JSONL. I worker si collegano alla
stessa run tramite un descrittore:

```python
descriptor = experiment.descriptor()

# Nel worker
worker_context = RunContext.attach(**descriptor)
worker_context.log_metrics(
    step=worker_step,
    metrics={"worker/throughput": throughput},
)
```

I worker scrivono eventi in SQLite WAL e non appendono direttamente ai file
JSONL. Quando il padre finalizza la run, ricostruisce gli export in ordine di
inserimento. Questo evita scritture concorrenti sullo stesso file.

Per riprendere una run nel processo proprietario:

```python
context = RunContext.attach(run_id, run_dir, root_dir, writer=True)
```

Solo il processo padre deve usare `writer=True`.

## Viste e integrazioni personalizzate

Una vista implementa un solo metodo:

```python
class MyView:
    def on_event(self, event):
        if event.kind == EventKind.METRICS:
            consume(event.payload)


with run("custom-view") as experiment:
    experiment.subscribe(MyView())
```

Gli errori sollevati da viste opzionali non interrompono il training e sono
consultabili in `experiment.listener_errors`.

Il salvataggio canonico avviene prima della notifica alle viste: una dashboard
difettosa non può causare la perdita della metrica.

## Creare un adapter di progetto

Un adapter deve tradurre concetti del dominio in API generiche, senza
modificare `ml_logger`.

Struttura raccomandata:

```text
project/
├── integrations/
│   └── project_logging.py
└── logger_config.yaml
```

Esempio:

```python
class DatasetRecorder:
    def __init__(self, context):
        self.context = context

    def save_split(self, path, split):
        return self.context.log_artifact(
            path,
            kind=f"dataset-{split}",
        )
```

L'adapter Regicide segue questo modello:

```python
from integrations.regicide_logging import GameRecorder
from ml_logger import run

with run("evaluation") as experiment:
    recorder = GameRecorder(experiment)
    ...
```

`GameCatalog`, `GameRecorder` e `serialize_game` appartengono all'adapter e non
al package generico.

## Indicazioni operative

- Creare una sola run per comando.
- Usare nomi metrici gerarchici come `train/loss` ed `eval/win_rate`.
- Registrare insieme le metriche dello stesso step.
- Lasciare dashboard e file aggregati al processo padre.
- Usare `log_artifact` per file che devono essere copiati nella run.
- Usare `save_result` per piccoli risultati JSON strutturati.
- Preferire `run()` a `start_run()` nel nuovo codice.
