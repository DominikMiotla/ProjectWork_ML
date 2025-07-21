# Classificazione NoVax con SBERT e Decision Tree

Questo script Python esegue la classificazione di testi relativi al COVID-19 utilizzando **Sentence-BERT** per generare embedding testuali e un **Decision Tree Classifier** per la classificazione. I parametri del classificatore vengono ottimizzati tramite **Grid Search con Cross Validation**.

## Dati di Input

Il dataset `new_Covid.csv` ha le seguenti features:

- `clean_text`: il testo preprocessato del documento
- `classificazione`: l'etichetta associata al testo (0 o 1)

## Come eseguire lo script

Posiziona `new_Covid.csv` nella stessa directory dello script.
Esegui lo script:
```python
python script.py
```
Durante l'esecuzione:

- Gli embedding verranno generati utilizzando il modello all-mpnet-base-v2 di SBERT.
- Gli embedding verranno salvati in cache nel file embeddings_NewDataset.joblib.
- Verrà effettuata una ricerca esaustiva su una griglia di iperparametri per ottimizzare le prestazioni del classificatore Decision Tree.
```python
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [4, 8, 10, 15],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [20, 50, 70]
}
```

# Migliori Iperparametri Trovati
```python
{
  'criterion': 'entropy',
  'max_depth': 8,
  'min_samples_leaf': 70,
  'min_samples_split': 10
}
```

## Risultati del Modello

Il classificatore finale è stato valutato su un test set separato (20% dei dati):

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.65      | 0.54   | 0.59     | 1957    |
| 1     | 0.75      | 0.82   | 0.79     | 3272    |

**Accuracy**: 0.72 (Total support: 5229)

**Macro avg**:
- Precision: 0.70
- Recall: 0.68
- F1-score: 0.69

**Weighted avg**:
- Precision: 0.71
- Recall: 0.72
- F1-score: 0.71

## Interpretazione dei Risultati

- Accuracy complessiva: 72%
- Il modello performa meglio nella classificazione della classe 1, con un F1-score di 0.79.
- La classe 0 ha performance inferiori, con un F1-score di 0.59,una maggiore difficoltà nella distinzione di questa classe.