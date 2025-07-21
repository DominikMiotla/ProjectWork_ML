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
Per la realizzazione del modello vengono utilizzate due features:
- group
- clean_text

Queste due features vengono concatenate per formare un’unica feature composita, che viene poi utilizzata come input per il modello PivotTree.

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
- Combina le *colonne group* e *clean_text* in una nuova feature **group_clean_text**
- Restituisce un DataFrame pronto per la fase di embedding


## Addestramento Modelli
Nel main script (model.py) vengono testate diverse configurazioni di PivotTree variando:
- max_depth (da 3 a 6)
- min_samples_leaf (1 o 2)

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

# Analisi Precision e Recall per la Classe 0

- **Precision e Recall sono molto bassi, specialmente il Recall (0.02–0.04):**  
  Significa che il modello raramente riconosce correttamente la classe 0.  
  L’aumento della profondità non porta un miglioramento sostanziale.

- **La precisione migliora solo marginalmente con profondità maggiori, ma resta bassa:**  
  Da 0.41 a un massimo di 0.57 (run 3–4), poi cala di nuovo → nessun trend chiaro di miglioramento.

- **Stesse prestazioni con `min_samples_leaf = 1` o `2`:**  
  Questo iperparametro non sembra influenzare molto i risultati per la classe 0.

- **Considerazioni:**  
  Potrebbe indicare che il problema non è la regolarizzazione, ma proprio la difficoltà intrinseca nel distinguere la classe 0 (forse per squilibrio o mancanza di segnali forti).  
  Ci concentreremo sulla classe 0, che ha un supporto minore.

---

## Risultati per classe 0

| RUN | Max_depth | min_samples_leaf | Precision | Recall |
|-----|-----------|------------------|-----------|--------|
| 1   | 3         | 1                | 0.41      | 0.02   |
| 2   | 3         | 2                | 0.41      | 0.02   |
| 3   | 4         | 1                | 0.57      | 0.04   |
| 4   | 4         | 2                | 0.57      | 0.04   |
| 5   | 5         | 1                | 0.46      | 0.04   |
| 6   | 5         | 2                | 0.46      | 0.04   |
| 7   | 6         | 1                | 0.45      | 0.04   |
| 8   | 6         | 2                | 0.47      | 0.04   |
