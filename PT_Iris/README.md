# Modello PivotTree sul Dataset Iris
In questo repository è implementato un modello PivotTree applicato al dataset Iris.

## Contenuto del repository
- `model_iris.py`: Script principale che esegue l’apprendimento del modello PivotTree sul dataset Iris.
- `PivotTree.py`: Implementazione dell’albero decisionale interpretabile basato su pivot (istanze di riferimento).
- `RuleTree.py`: Implementazione base di un albero decisionale tradizionale.
- `Utilis.py`:Funzioni di supporto per analisi, visualizzazione e interpretazione dei modelli PivotTree.

- `albero_con_pivot.txt`: File di testo contenente la struttura dell’albero decisionale con pivot discriminativi e descrizioni. Generato dalla funzione `visualize_tree_with_pivots`.
- `info_fit.txt`:Report sintetico con informazioni sul dataset e dettagli dell’addestramento, prodotto dallo script `model_iris.py`, attraverso ridirezione dello stdout.
- `valutazione_modello.txt`: Report sulle performance del modello, generato dalla funzione `report_modello`.

#### Dettagli sul file info_fit.txt
Il `file info_fit.txt` contiene:
- Informazioni generali sul dataset Iris
- Descrizione delle classi presenti
- Dettagli sul modello addestrato e sulle metriche di valutazione ottenute

# Come eseguire lo script
Esegui da terminale: `python model_iris.py`

## Cosa fa lo script
- Caricamento dataset Iris: carica i dati e stampa informazioni sulle istanze, caratteristiche, classi e verifica la presenza di valori nulli.
- Visualizzazione delle prime righe: mostra una tabella riassuntiva delle prime 5 istanze.
- Suddivisione dei dati: separa il dataset in training (70%) e test (30%).
- Creazione e addestramento modello PivotTree: con parametri specifici `(max_depth=4, min_samples_leaf=3,...)`.

- Valutazione del modello: genera un report delle performance sul set di test.

- Visualizzazione dell’albero decisionale: stampa la struttura dell’albero basata sui pivot.

- Mostra il percorso decisionale: visualizza come viene classificata una specifica istanza di test (indice 2).

## File prodotti e output
- File di valutazione generato dalla funzione report_modello (solitamente valutazione_modello.txt).

- Visualizzazione testuale dell’albero decisionale con pivot.

- Percorso decisionale per l’istanza di test indicata.

## Personalizzazioni
Puoi modificare i parametri del modello PivotTree nel blocco di creazione, ad esempio:
```python
pt = PivotTree(
    max_depth=4,
    min_samples_leaf=3,
    model_type='clf',
    pairwise_metric='euclidean',
    allow_oblique_splits=True
)
```

# Valutazione del modello

Il modello ha ottenuto ottime performance durante la fase di valutazione, dimostrando un'elevata capacità di classificazione.  

- **Accuracy**: 100%, il modello ha classificato correttamente tutte le istanze del test set.
- **Confusion Matrix**: mostra una perfetta classificazione per tutte e tre le classi (0, 1 e 2), senza errori di classificazione.

| Pred\True |     0     |     1     |     2     |
|:---------:|:---------:|:---------:|:---------:|
|     0     |    19     |     0     |     0     |
|     1     |     0     |    13     |     0     |
|     2     |     0     |     0     |    13     |

- **Classification Report**:  
  - Precision, recall e F1-score sono tutti pari a 1.00 per ciascuna classe, confermando la completa accuratezza e affidabilità del modello su ogni categoria.
  - Support: il numero di campioni per ogni classe è rispettivamente 19, 13 e 13.

In sintesi, il modello ha raggiunto performance eccellenti, classificando con precisione tutte le osservazioni nel dataset di test, senza errori.
