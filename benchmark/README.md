# Benchmark QASPER

Protocollo congelato: vedi `thesis/chapters/03-methodology.tex`. I comandi qui
sotto sono quelli con cui il golden set è stato fissato, e non vanno cambiati
durante la campagna sperimentale.

## Dataset

- ablation (dev): `qasper-train-dev-v0.3.tgz`
- run finale (test) + evaluator ufficiale: `qasper-test-and-evaluator-v0.3.tgz`

Entrambi da `https://qasper-dataset.s3.us-west-2.amazonaws.com/`. I file JSON non
sono versionati qui: il golden set si rigenera dal comando sotto, che è
deterministico a parità di seed.

## Golden set congelato

```
python -m benchmark.build_golden_set qasper-dev-v0.3.json \
    --out benchmark_out --papers 100 --seed 42 --pdf-dir benchmark_out/pdfs
```

Risultato: **100 paper, 218 domande**, di cui 90 (41,3%) a evidenza multipla.
Esclusioni sul dev completo (1005 domande): 237 abstractive, 113 yes/no,
99 unanswerable, 20 con evidenza in tabelle o figure. 1 PDF non scaricabile
(`1802.00396`).

## Allineamento evidenza/chunk

Richiede GROBID attivo (`docker-compose up`).

```
python -m benchmark.extract_chunks --pdf-dir benchmark_out/pdfs --out benchmark_out/chunks
python -m benchmark.align_evidence benchmark_out/golden_set.jsonl \
    --chunks benchmark_out/chunks --out benchmark_out --sample 30 --seed 42
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

Risultato su QASPER (100 paper): allineamento paragrafi 52,4%, allineamento
domande 55,1% (120/218, soglia 0,8). Il calo rispetto alle iterazioni
precedenti è atteso: molti match che sembravano corretti erano evidenza
sparsa su più chunk (voci di lista separate), ora correttamente esclusi
perché un retriever che recupera un chunk alla volta non li darebbe comunque
per intero. Tasso d'errore sul campione di validazione a mano (n=30, seed
42), col criterio definitivo: **0% (0/30)**.

Validazione della soglia senza GROBID (usa i paragrafi di QASPER come se fossero
i nostri chunk, quindi misura i soli falsi negativi del criterio):

```
QASPER_DEV=qasper-dev-v0.3.json pytest tests/test_align_evidence.py
```

## Golden set QASA

Secondo dataset passage-level, stessa infrastruttura di allineamento. Richiede
un release JSON di QASA (`testset_answerable_1554_v1.1.json` da
`https://github.com/lgresearch/QASA`, MIT license). A differenza di QASPER,
QASA non usa id arXiv come chiave: ogni domanda ha già la sua evidenza
(`evidential_info[].context`), quindi non c'è filtro per tipo di risposta,
solo deduplica. Il download risolve titolo -> id arXiv via l'API di ricerca
arXiv, accettando solo un match esatto sul titolo normalizzato; i paper non
risolti sono riportati e vanno recuperati a mano.

```
python -m benchmark.build_qasa_golden_set testset_answerable_1554_v1.1.json \
    --out benchmark_out_qasa --papers 100 --seed 42 --pdf-dir benchmark_out_qasa/pdfs
```

Risultato: **100 paper, 1.381 domande**, di cui 631 (45,7%) a evidenza multipla
(1 domanda esclusa per evidenza vuota). PDF: 90/100 scaricati, 9 titoli non
risolti su arXiv (verosimilmente artefatti di estrazione nel campo `title` di
QASA, es. sillabazione spezzata), 1 download fallito.

L'allineamento evidenza/chunk riusa `extract_chunks.py` e `align_evidence.py`
senza modifiche specifiche a QASA, puntati su `benchmark_out_qasa/`.
Risultato: allineamento paragrafi 61,2%, allineamento domande 43,7%
(604/1381, soglia 0,8; 145 domande escluse per assenza di chunk sui 10 paper
non scaricati). Il tasso è più basso che su QASPER perché il campo `context`
di QASA concatena punti elenco distinti più spesso — esattamente il caso di
evidenza multi-chunk che il criterio ora esclude per scelta. Tasso d'errore
sul campione di validazione a mano (n=30, seed 42), col criterio definitivo:
**0% (0/30)**.

## Scoring

`qasper_evaluator.py` è lo script ufficiale, copiato senza modifiche. Va usato
sul `golden_gold.json` prodotto dal builder, non sul rilascio completo, altrimenti
conta come predizioni mancanti tutte le domande escluse dal golden set.

```
python benchmark/qasper_evaluator.py \
    --predictions predictions.jsonl \
    --gold benchmark_out/golden_gold.json \
    --text_evidence_only
```

Formato delle predizioni, una riga per domanda:
`{"question_id": ..., "predicted_answer": ..., "predicted_evidence": [...]}`

Recall@k e MRR non sono coperti dallo script ufficiale e vanno calcolati a parte,
sui risultati intermedi del retrieval.
