# No-Vax Classification with PivotTree

Questo progetto ha lo scopo di **addestrare un modello PivotTree** per classificare commenti social in due categorie:
- **Classe 0** → Non No-Vax
- **Classe 1** → No-Vax

Il dataset contiene **commenti in lingua italiana**, raccolti da post sui social.
# Dataset Info
Il dataset di questo repository è stato ampliato mediante tecniche di data augmentation, utilizzando un Large Language Model (LLM). Per maggiori dettagli sulle modifiche apportate al dataset, si rimanda al seguente repository, dove il processo è descritto in modo approfondito:
https://github.com/DominikMiotla/ProjectWork_ML/tree/devel/DataAugmentation_LLM

# Struttura del Repository
- Dataset/
    - `new_Covid.csv`: Dataset
- output/
    - `embeddings_NewDataset.joblib`: cache embeddings
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
- Generazione degli embeddings tramite SBERT
- Visualizzazione degli embeddings (t-SNE)
- Addestramento di più modelli PivotTree (con varie profondità e foglie minime)
- Salvataggio dei report in output/



## Addestramento Modelli
Nel main script (model.py) vengono testate diverse configurazioni di PivotTree variando:
- max_depth [3,4,5,6]
- min_samples_leaf [10,20,30,50]

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

# Valutazione Modelli di Classificazione 

## Panoramica del Dataset

- Il dataset presenta **classi sbilanciate**, con la **classe 0 rappresentata al 38%**.
- Per questo motivo, la **semplice Accuracy non è una metrica sufficiente**.
- Si è posta particolare attenzione a:
  - **Recall e Precision per la classe 0** (minoritaria)
  - Parametri `max_depth` e `min_samples_leaf` dei modelli

---

## Parametri testati

| RUN | max_depth | min_samples_leaf | Accuracy | Precision_0 | Recall_0 | Precision_1 | Recall_1 |
|-----|-----------|------------------|----------|-------------|----------|--------------|----------|
| 1   | 3         | 10               | 0.7286   | 0.66        | 0.59     | 0.76         | 0.81     |
| 5   | 4         | 10               | 0.7344   | 0.68        | 0.57     | 0.76         | 0.83     |
| 9   | 5         | 10               | 0.7328   | 0.76        | 0.44     | 0.73         | 0.91     |
| 13  | 6         | 10               | 0.7328   | 0.76        | 0.44     | 0.73         | 0.91     |

*Tabella sintetica di alcuni modelli rappresentativi*

---

## Modelli consigliati

Modelli con **`max_depth = 3–4`** offrono un buon bilanciamento tra le classi:

- **RUN 5 (max_depth=4, min_samples_leaf=10)**
  - Precision_0: 0.68
  - Recall_0: 0.57
  - Precision_1: 0.76
  - Recall_1: 0.83


---

## Modelli da evitare

Modelli con **`max_depth ≥ 5`** presentano un Recall_0 troppo basso:

- Esempio: RUN 9 (max_depth=5)
  - Recall_0: **0.44**
  - Recall_1: **0.91**
  -  Modello troppo sbilanciato verso la classe 1