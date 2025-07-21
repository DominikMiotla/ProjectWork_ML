# Classificazione NoVax K-Nearest Neighbors e Sentence-BERT

Questo progetto implementa un sistema di classificazione testuale utilizzando embeddings generati tramite **Sentence-BERT (SBERT)** e un classificatore **K-Nearest Neighbors (KNN)** ottimizzato con ricerca a griglia (GridSearchCV).

## Descrizione

Il codice esegue i seguenti passaggi:

1. Caricamento di un dataset CSV contenente testi puliti (`clean_text`) e le rispettive classi (`classificazione`).
2. Suddivisione del dataset in train e test (80% train, 20% test) con stratificazione.
3. Generazione degli embeddings dei testi usando il modello SBERT `all-mpnet-base-v2`. 
   - Gli embeddings vengono salvati in cache (`embeddings_NewDataset.joblib`) per evitare di rigenerarli.
4. Addestramento di un classificatore KNN con ottimizzazione degli iperparametri (`n_neighbors`, `weights`, `metric`) tramite GridSearchCV con validazione incrociata a 5 fold.
5. Valutazione finale del modello con un report di classificazione sul set di test.

```python
param_grid = {
    'n_neighbors': [3,4, 5, 10,20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

# Esegui lo script:
```python
python nome_script.py
```

Lo script stamperà:
- I migliori iperparametri trovati da GridSearchCV.
- Il report di classificazione sul set di test.

# Valutazione del modello
**Migliori iperparametri:**  
```python
{'metric': 'euclidean', 'n_neighbors': 10, 'weights': 'distance'}
```
| Classe | Precision | Recall | F1-score | Supporto |
|--------|-----------|--------|----------|----------|
| 0      | 0.78      | 0.69   | 0.73     | 1957     |
| 1      | 0.83      | 0.88   | 0.85     | 3272     |

Il modello mostra buone prestazioni complessive, con un F1-score di 0.73 per la classe 0 e 0.85 per la classe 1. Questo indica che il modello è più efficace nel riconoscere la classe 1, come evidenziato dall’elevato valore di recall (0.88) e precision (0.83).

La classe 0 presenta una performance leggermente inferiore, con un recall più basso (0.69), suggerendo che il modello fatica un po’ a identificare correttamente tutti i campioni appartenenti a questa classe. Tuttavia, la precisione rimane discreta (0.78), indicando che quando il modello predice la classe 0, lo fa con una buona accuratezza.

Nel complesso, il modello sembra bilanciato e più performante nel riconoscimento della classe maggioritaria (classe 1), mentre potrebbe essere migliorato nell’identificazione della classe minoritaria (classe 0).