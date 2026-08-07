# Benchmark

Protocollo: vedi `thesis/chapters/03-methodology.tex`. I comandi qui sotto sono
quelli con cui i set sono stati fissati, e non vanno cambiati durante la campagna.

## Layout

```
data/raw/                      release sorgente (non versionate)
data/<split>/<corpus>/         golden set, pdfs/, chunks/, allineamento, pdf_hash_map.json
data/<split>/grobid/           cache GROBID e PDF dello split (condivisa dai due corpora)
runs/dev/<campagna>/<corpus>/  risultati delle campagne sullo sviluppo
runs/heldout/                  la run finale, una sola
runs/archive/                  le campagne dello sviluppo storico (output_*), intatte
```

`<split>` è `dev` o `heldout`, `<corpus>` è `qasper` o `qasa`. Tutto è ignorato da git.

## Dataset

- QASPER: `qasper-train-dev-v0.3.tgz` e `qasper-test-and-evaluator-v0.3.tgz` da
  `https://qasper-dataset.s3.us-west-2.amazonaws.com/`.
- QASA: `testset_answerable_1554_v1.1.json` da `https://github.com/lgresearch/QASA`
  (`data/`, MIT license).

Secondo dataset passage-level, stessa infrastruttura di allineamento. Richiede
un release JSON di QASA (`testset_answerable_1554_v1.1.json` da
`https://github.com/lgresearch/QASA`, MIT license). A differenza di QASPER,
QASA non usa id arXiv come chiave: ogni domanda ha già la sua evidenza
(`evidential_info[].context`), quindi non c'è filtro per tipo di risposta,
solo deduplica. Il download risolve titolo -> id arXiv via l'API di ricerca
arXiv, accettando solo un match esatto sul titolo normalizzato; i paper non
risolti sono riportati e vanno recuperati a mano.

## Sviluppo e held-out

Split 80/20 sui paper, con QASPER e QASA in proporzione tale da avere un numero
simile di domande allineate (QASA ne ha circa 7 per paper, QASPER circa 1). I paper
sono campionati solo tra quelli con un PDF già scaricato (`--pdf-pool`), così le
dimensioni fissate sono quelle indicizzate.

```
A=runs/archive
python -m benchmark.build_qasper_golden_set $A/output_qasper/golden_gold.json \
    $A/output_qasper_test/golden_gold.json --pdf-pool $A/output_qasper/pdfs \
    $A/output_qasper_test/pdfs --papers 240 --out-dir data/dev/qasper
python -m benchmark.build_qasper_golden_set data/raw/qasper-dev-v0.3.json \
    --exclude-papers $A/output_qasper/golden_set.jsonl \
    --pdf-pool $A/output_final_qasper/pdfs --papers 60 --out-dir data/heldout/qasper
python -m benchmark.build_qasa_golden_set data/raw/testset_answerable_1554_v1.1.json \
    --pdf-pool $A/output_qasa/pdfs --papers 40 --out-dir data/dev/qasa
python -m benchmark.build_qasa_golden_set data/raw/testset_answerable_1554_v1.1.json \
    --exclude-papers $A/output_qasa/golden_set.jsonl \
    --pdf-pool $A/output_final_qasa/pdfs --papers 10 --out-dir data/heldout/qasa
```

| Set | Paper | Domande | Allineate |
|---|---|---|---|
| dev QASPER | 240 | 519 | 243 (46,8%) |
| dev QASA | 40 | 564 | 269 (47,7%) |
| held-out QASPER | 60 | 132 | 62 (47,0%) |
| held-out QASA | 10 | 148 | 83 (56,1%) |

Lo sviluppo QASPER è tratto dai 450 paper già usati dallo sviluppo storico (dev e
test split), lo sviluppo QASA dai 90 con PDF del vecchio sviluppo. L'held-out è
fatto solo di paper che nessuna campagna ha mai eseguito: per QASPER 60 dei 137
paper del dev split rimasti fuori dallo sviluppo storico, per QASA 10 degli 11
mai usati con un PDF (QASA pubblica 112 paper; `paper_22` non è su arXiv). Le
domande QASA sono correlate dentro il paper (269 domande da 39 paper), quindi la
loro metà vale meno del suo n.

**L'held-out va usato una volta sola**, con la configurazione consegnata: rifarlo
dopo aver guardato i risultati lo trasforma in un secondo set di sviluppo.

## Allineamento evidenza/chunk

Richiede GROBID attivo (`docker compose up -d grobid`). Per ogni set:

```
python -m benchmark.extract_chunks --pdf-dir data/dev/qasper/pdfs \
    --out-dir data/dev/qasper/chunks --work-dir data/dev/grobid
python -m benchmark.align_evidence data/dev/qasper/golden_set.jsonl \
    --chunks-dir data/dev/qasper/chunks --out-dir data/dev/qasper --sample 30 --seed 42
```

Criterio: un paragrafo è allineato solo quando un **singolo** chunk lo copre.
GROBID segmenta per paragrafo/voce di lista, quindi un'evidenza QASPER/QASA
che concatena più punti elenco distinti in un'unica annotazione, ma che nel
PDF corrisponde a più chunk separati, resta volutamente non allineata:
recuperare uno solo di quei chunk non darebbe comunque accesso all'evidenza
completa. Il punteggio è token condivisi diviso la lunghezza del testo **più
lungo** tra paragrafo e chunk (non quello più corto), soglia 0,8 — equivale a
richiedere sia che il chunk copra la maggior parte del paragrafo (recall) sia
che il paragrafo sia la maggior parte del chunk (precisione), in un'unica
formula. I paragrafi sotto i 10 token non si allineano e si contano a parte:
sono frammenti di una riga, spesso attorno a una formula, che nessuna soglia
può collocare in modo affidabile. Le domande la cui evidenza non è allineata
*per intero* finiscono in `alignment_dropped.jsonl`, mai in un abbinamento
approssimato.

Il criterio è passato per due iterazioni prima di questa, entrambe emerse
dalla validazione a mano (vedi cronologia in `alignment_manual_check.md` e
nel changelog del progetto): normalizzare sul testo più corto lasciava
passare chunk troncati (coprivano solo una parte del paragrafo, ma il
punteggio saturava vicino a 1,0) e chunk troppo grossolani (paragrafo
affogato in un blocco che fonde contenuti non correlati); un tentativo
intermedio accettava l'unione di più chunk per gestire il caso di un
paragrafo diviso, ma nei dati reali quella divisione è sempre tra voci di
lista distinte, non un taglio artificiale a metà frase — quindi si è deciso
di richiedere un chunk singolo, senza unione.

`alignment_manual_check.md` è il campione da validare a mano: la quota di coppie
sbagliate è il tasso di errore da riportare in tesi accanto al tasso di
allineamento. `align_evidence.py` non sovrascrive il file se esiste già (serve
`--force-sample` per rigenerarlo), per non perdere una validazione già fatta.

Sullo sviluppo storico il tasso d'errore del campione di validazione a mano
(n=30, seed 42) è stato **0% (0/30)** su QASPER e su QASA, e 0/30 anche sui set
finali da cui l'held-out è tratto.

## Indicizzazione (`index_pdfs.py`)

Le metriche di retrieval richiedono che i PDF del set siano dentro Qdrant.
`index_pdfs.py` li ingesta come sorgente cartella, bypassando Zotero, e scrive la
mappa `{paper_id: pdf_hash}` che lo scoring usa per legare le domande ai chunk
indicizzati. Richiede GROBID e Qdrant attivi.

```
python -m benchmark.index_pdfs --pdf-dir data/dev/qasper/pdfs \
    --out-file data/dev/qasper/pdf_hash_map.json --work-dir data/dev/grobid \
    --qdrant-collection-suffix _dev
```

Una collection per split (`_dev`, `_heldout`), con i due corpora dentro: il
retrieval è su tutta la libreria e una libreria contiene paper di entrambi, quindi
sviluppo e held-out si interrogano allo stesso modo. Due split non devono mai
condividere una collection.

La contestualizzazione dei chunk è stata misurata sul dev con un indice a parte
(`_dev_ctx`, `runs/dev/contextualization/`), non ha mosso nessuna metrica e ora
non c'è più: il codice che l'ha prodotta sta nella history.

## Campagne

```
python -m benchmark.ablation --golden-dir data/dev/qasper \
    --hash-map data/dev/qasper/pdf_hash_map.json --work-dir data/dev/grobid \
    --out-file runs/dev/ablation/qasper/ablation_results.csv \
    --strata-file runs/dev/ablation/qasper/ablation_by_stratum.md --collection-suffix _dev
```

Stesso comando con `qasa` per l'altro corpus. Senza `--only` gira la griglia
intera più i bracci incrociati; i CSV si riprendono dalle configurazioni già
scritte.

## Sviluppo storico (`runs/archive/`)

Le campagne delle fasi precedenti girano su set diversi e restano dove erano:
QASPER 100 paper del dev split (218 domande, 120 allineate), QASA 100 paper (90
PDF, 1.381 domande, 604 allineate), held-out del cap. 8 su 350 paper del test
split (740 domande, 340 allineate), e i set finali `output_final_*` e
`output_heldout_*` mai eseguiti. Ogni collection era per corpus
(`_qasper`, `_qasa`, `_qasper_test`). I numeri della tesi che vengono da lì
citano quei path.

## Scoring

`qasper_evaluator.py` è lo script ufficiale. Rispetto al rilascio ha solo
riordino degli import e due riscritture equivalenti (`max(...)` al posto di
`sorted(..., reverse=True)[0]`, che su ex aequo restituisce lo stesso elemento, e
`with open`): la semantica di punteggio è invariata. Va usato
sul `golden_gold.json` prodotto dal builder, non sul rilascio completo, altrimenti
conta come predizioni mancanti tutte le domande escluse dal golden set.

```
python benchmark/qasper_evaluator.py \
    --predictions predictions.jsonl \
    --gold data/dev/qasper/golden_gold.json \
    --text_evidence_only
```

Formato delle predizioni, una riga per domanda:
`{"question_id": ..., "predicted_answer": ..., "predicted_evidence": [...]}`

Recall@k e MRR non sono coperti dallo script ufficiale e vanno calcolati a parte,
sui risultati intermedi del retrieval.

### Metriche proprie (`retrieval_metrics.py`)

| Metrica | Cosa risponde | Perché non basta lo script ufficiale |
|---|---|---|
| `recall@k` | il retrieval trova i chunk gold? | assente dallo scorer QASPER |
| `mrr` | quanto in alto li mette? | assente |
| `evidence_precision/recall/f1` | l'attribuzione (highlighting) è corretta? | l'Evidence F1 ufficiale confronta stringhe esatte contro testo LaTeX che i paragrafi GROBID non eguagliano mai: vale 0.0 per qualsiasi configurazione |
| `bootstrap_ci` | la differenza tra due run è reale? | su 120 domande allineate un punto di differenza è rumore |

`precision@k` è stata usata e poi rimossa. Era stata aggiunta per rendere
misurabili i parametri di filtro, e non lo faceva: era calcolata sulla stessa
lista `ranked_ids` di `recall@k`, quindi vedeva gli stessi tagli e nessun altro.
In più, con mediana di un chunk gold per domanda (89 su 120 in QASPER), vale
`recall@k * |gold| / k` — una riscalatura, non una seconda misura. Il ruolo per
cui era stata introdotta lo svolgono le metriche di attribuzione, che valutano
ciò a cui il sistema si è impegnato. Le colonne restano nei CSV e nei JSONL già
prodotti, che non vengono riscritti.

`evidence_*` valuta gli id `(pdf_hash, chunk_index)` che il sistema ha
effettivamente attribuito alla risposta, non le stringhe: è la controparte
misurabile dell'Evidence F1 ufficiale, resa possibile dall'allineamento fuzzy
di `align_evidence.py`.

### Stratificazione (`stratify.py`)

I due golden set differiscono sistematicamente (vedi `compare_datasets.py`):
una media aggregata non può dire *perché* un dataset va peggio. Ogni metrica è
quindi riportata anche per forma della domanda, numero di evidenze e
dispersione delle evidenze nel documento. Gli strati sotto le 20 domande sono
marcati come non interpretabili anziché essere taciuti.

### Confronto tra i due dataset (`compare_datasets.py`)

```
python -m benchmark.compare_datasets \
    --dataset QASPER=data/dev/qasper --dataset QASA=data/dev/qasa \
    --out-file runs/dev/dataset_comparison.md
```

Nota: QASA non ha risposte brevi annotate, quindi l'Answer F1 non è calcolabile
su quel dataset. QASPER copre retrieval + answer + evidence, QASA solo
retrieval + evidence.
