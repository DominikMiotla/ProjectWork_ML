# PivotTree Sentiment Classification su IMDB
Questo progetto utilizza il dataset IMDB Reviews per eseguire una classificazione di sentimenti *(positivo/negativo)* tramite il modello PivotTree. Le recensioni vengono convertite in embedding vettoriali tramite SBERT, e successivamente classificate con PivotTree.

## Caratteristiche Principali
- Caricamento e caching automatico del dataset IMDB
- Generazione e caching degli embedding SBERT (all-mpnet-base-v2)
- Addestramento di un albero decisionale custom (PivotTree)
- Visualizzazione dell’albero con i pivot
- Tracciamento del path decisionale per un esempio specifico
- Generazione di un report di valutazione su test set

# Struttura del progetto
PT_IMDB/
- models_imdb.py: Fit di un ensable di PivotTree
- model_imdb.py: Fit di un PivotTree
- tensorflow_datasets: Directory per cache dei dati e embedding
   -  imdb_dataset.joblib
   - embeddings.joblib
- `RuleTree.py` Implementazione di base di un **albero decisionale**.
  
- `PivotTree.py` Costruisce un **albero decisionale interpretabile** utilizzando istanze pivot *(esempi di riferimento)*.
  
- `Utilis.py` Fornisce **funzioni di supporto** sviluppate per analizzare, visualizzare e interpretare i modelli PivotTree.

# Script models_imdb.py
Questo script esegue una Grid Search per addestrare e valutare il modello PivotTree sul dataset di recensioni IMDb, utilizzando Sentence Embeddings generati dal modello all-mpnet-base-v2 di SentenceTransformers.
 
## Sequenza delle operazioni
- Scarica il dataset IMDb (se non è già salvato in cache)
- Estrae i testi e le etichette *(positivo/negativo)*.
- Genera gli embeddings dei testi con SBERT (all-mpnet-base-v2).
- Esegue una Grid Search con combinazioni di iperparametri per addestrare un classificatore PivotTree.
- Valuta l'accuracy su ogni combinazione.
- Salva un file di report pivot_tree_report.txt con i parametri e l'accuracy per ciascun modello.

### Parametri della Grid Search
Il dizionario PARAM_GRID contiene i seguenti iperparametri:

```python
PARAM_GRID = {
    "max_depth": [3, 4, 5, 6],
    "min_samples_leaf": [2, 3, 5, 8],
    "allow_oblique_splits": [False, True]
}
```
Spiegazione:
- max_depth: profondità massima dell’albero.
- min_samples_leaf: numero minimo di campioni in una foglia.
- allow_oblique_splits: consente split obliqui (True/False).

Questi parametri vengono combinati tramite itertools.product, producendo 32 combinazioni in totale (4 x 4 x 2)

### File di output: pivot_tree_report.txt
Per ogni combinazione di parametri, il file di report conterrà una riga come questa:

```python
[AAAA-MM-DD HH:MM:SS] Params: max_depth-4_min_samples_leaf-2_allow_oblique_splits-True | Accuracy: Val
```
Ogni riga rappresenta:
- Timestamp dell’esperimento
- Parametri usati
- Accuracy ottenuta sul test set

# Script model_imdb.py
Questo script costruisce un classificatore di sentiment basato su PivotTree applicato al dataset IMDB Reviews. Il flusso include il download e preprocessing del dataset, generazione degli embedding tramite Sentence-BERT, addestramento del modello, e valutazione delle performance.



Parametri del modello:
- `max_depth=6`	Profondità massima dell'albero decisionale
- `min_samples_leaf=3`	Minimo numero di campioni in una foglia
- `model_type='clf'`	Tipo di modello (classificatore)
- `pairwise_metric='euclidean'`	Metodologia per calcolare la distanza tra punti (metrica euclidea)
- `allow_oblique_splits=True` Consente split obliqui (basati su più pivot)

## Valutazione
- Accuracy calcolata su dati di test
- Report dettagliato salvato su `Output/valutazione_modello.txt`

## Visualizzazione
- Struttura dell’albero salvata su `Output/albero_con_pivot.txt`
- Percorso decisionale per un'istanza di test (indice 2) mostrato a terminale
