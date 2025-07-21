# No-Vax Classification with PivotTree

Questo progetto ha lo scopo di **addestrare un modello PivotTree** per classificare commenti social in due categorie:
- **Classe 0** → Non No-Vax
- **Classe 1** → No-Vax

Il dataset contiene **commenti in lingua italiana**, raccolti da post sui social.
# Dataset Info
Numero totale di istanze: 19.104

Colonne (features):
- id
- date
- group
- pre_clean_text
- clean_text
- classificazione : label assegnata all'istanza

Numero di valori unici nella feature group: 12

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19104 entries, 0 to 19103
Data columns (total 6 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   id               19104 non-null  int64 
 1   date             19104 non-null  object
 2   group            19104 non-null  object
 3   pre_clean_text   19104 non-null  object
 4   clean_text       19104 non-null  object
 5   classificazione  19094 non-null  object
dtypes: int64(1), object(5)
memory usage: 895.6+ KB
```

# Feature Engineering

Per la realizzazione del modello viene utilizzata la feature:
- clean_text

Dopo la generazione degli embeddings, viene eseguita una procedura di **data augmentation** sulla classe 0, che risulta essere sottorappresentata nel dataset.


# Struttura del Repository
- Dataset/
    - `post_cleaned_it_only_GEMMA3_finale.csv`: Dataset
- output/
    - `embeddings_Group_clean_text.joblib`: cache embeddings
    -  `info_dataset.txt`: info su preprecessing dataset 
    - `Report_MD{depth}_MSL{leaf}.txt`: report per modello addestrato
    - `tsne_embeddings_train.png`: visualizzazione istanze dopo embeddings 
- `model.py`: Main script di esecuzione
- `RuleTree.py` Implementazione di base di un albero decisionale.
- `PivotTree.py` Costruisce un albero decisionale interpretabile utilizzando istanze pivot (esempi di riferimento).
- `Utilis.py` Fornisce funzioni di supporto sviluppate per analizzare, visualizzare e interpretare i modelli PivotTree.

# Esecuzione
```python
python model.py
```
## Work Flow
- Preprocessing del dataset
- Generazione degli embeddings tramite SBERT
- Visualizzazione degli embeddings (t-SNE)
- Addestramento di più modelli PivotTree (con varie profondità e foglie minime)
- Salvataggio dei report in output/

# Dettagli del Preprocessing (preprocessingDataset)
La funzione *preprocessingDataset(file_path)* svolge un ruolo chiave nella pulizia e preparazione del dataset. In dettaglio:
- Legge il file CSV e crea un report dettagliato in *output/info_dataset.txt*
- Corregge errori nelle etichette della colonna classificazione *(es. spazi, punti, minuscole)*
Mantiene solo le classi YES e NO, mappandole in:
    - YES → 1 (No-Vax)
    - NO → 0 (Non No-Vax)
- Mantine solo la *features *clean_text* 
- Restituisce un DataFrame pronto per la fase di embedding


## Addestramento Modelli
Nel main script (model.py) vengono testate diverse configurazioni di PivotTree variando:
- max_depth (da 3 a 12)
- min_samples_leaf (1 o 5)

Per ogni combinazione viene generato un report di classificazione salvato nella cartella output/.

## Esempio di Report Generato
Ogni file `Report_MD{X}_MSL{Y}.txt` contiene:
- Precision, recall, F1-score
- Matrice di confusione
- Accuracy

# Embedding Testi (get_embeddings)
Per rappresentare le frasi testuali numericamente, viene utilizzato SentenceTransformer('all-mpnet-base-v2'). La funzione: Controlla se esiste già un file cache `embeddings_Group_clean_text.joblib`
- Se sì, lo carica per evitare computazioni ripetute
- Altrimenti:
    - Scarica il modello SBERT
    - Genera embeddings per i testi di train e test
    - Salva gli embeddings in cache per usi futuri

## Visualizzazione
La funzione `plot_embeddings` usa t-SNE per proiettare gli embeddings in 2D e salvare una mappa (output/tsne_embeddings_train.png) utile per analizzare la separabilità tra le classi.

# Valutazione dei Modelli per la Classe Minoritaria (Classe 0)


## Sintesi delle Performance

| RUN     | max_depth | min_samples_leaf | Precision | Recall | F1-score |
|---------|------------|------------------|-----------|--------|----------|
| 1-4     | 3          | 10-50            | 0.27      | 0.22   | 0.242    |
| 5-7     | 4          | 10-30            | 0.30      | 0.21   | 0.247    |
| 8       | 4          | 50               | 0.29      | 0.21   | 0.244    |
| 9-11    | 5          | 10-30            | 0.25      | 0.36   | 0.295    |
| 12      | 5          | 50               | 0.25      | 0.37   | 0.298    |
| 13-15   | 6          | 10-30            | 0.25      | 0.36   | 0.295    |
| 16      | 6          | 50               | 0.25      | 0.37   | 0.298    |

### Trade-off Precision-Recall

#### Alberi poco profondi (`max_depth = 3-4`)
- **Precisione alta**: fino a **0.30**
- **Recall basso**: circa **0.21-0.22**

#### Alberi più profondi (`max_depth = 5-6`)
- **Recall più alto**: fino a **0.37** 
- **Precisione più bassa**: stabile a **0.25**
-  Migliore per identificare la classe rara

### Impatto degli Iperparametri

- **`max_depth` (profondità dell'albero)**:
  - Maggiore profondità = **+Recall**, **−Precision**
  - Alberi profondi catturano pattern rari.

- **`min_samples_leaf` (dimensione minima foglie)**:
  - Valori alti (es. 50) = **modelli più stabili**
  - Aumentare da 30 → 50 **migliora leggermente il recall** (0.36 → 0.37) **senza penalizzare la precisione**

