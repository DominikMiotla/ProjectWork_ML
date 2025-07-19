# Pivot Tree
Questo repository include le implementazioni di **RuleTree** e **PivotTree**, come descritto nell’articolo *"Data-Agnostic Pivotal Instances Selection for Decision-Making Models".*
 L’obiettivo è apprendere un modello predittivo gerarchico e interpretabile.



## File presenti nel repository

- `RuleTree.py` Implementazione di base di un **albero decisionale**.
  
- `PivotTree.py` Costruisce un **albero decisionale interpretabile** utilizzando istanze pivot *(esempi di riferimento)*.
  
- `Utilis.py` Fornisce **funzioni di supporto** sviluppate per analizzare, visualizzare e interpretare i modelli PivotTree.



## Come addestrare un Pivot Tree

Esempio di addestramento di un `PivotTree`:

```python
from PivotTree import PivotTree

# Dati di trainig
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Pivot Tree 
pt = PivotTree(
    max_depth=3,
    min_samples_leaf=2,
    model_type='clf',
    pairwise_metric='euclidean',
    verbose=True
)
pt.fit(X_train, y_train)
```
### Parametri principali di PivotTree
- `max_depth`: imposta la profondità massima dell’albero, controllando la complessità del modello.

- `min_samples_leaf`: definisce il numero minimo di campioni richiesti in una foglia per permettere una suddivisione.

- `model_type`: specifica il tipo di modello; `clf` indica un classificatore.

- `pairwise_metric`: la metrica utilizzata per calcolare la distanza tra le istanze pivot, ad esempio `euclidean` per la distanza euclidea.

- `verbose`: se impostato a True, mostra informazioni dettagliate durante la costruzione dell’albero per facilitare il debug e il monitoraggio.


## Utility per Modelli PivotTree
Questo modulo fornisce funzioni di supporto per l’analisi, la visualizzazione e la valutazione di modelli basati su PivotTree.

Le principali funzionalità includono:

- Visualizzazione tabellare delle prime istanze di un dataset.

- Valutazione completa del modello con salvataggio di metriche e report.

- Estrazione e visualizzazione dei pivot discriminativi e descrittivi da un albero PivotTree.

- Visualizzazione del percorso decisionale di singoli campioni.

### Funzioni
- `print_instances_table(X, feature_names, n=5)`
Stampa una tabella formattata con le prime n istanze e i valori delle feature. **Parametri:**

    - `X`: array/lista delle istanze.

    - `feature_names`: lista dei nomi delle feature.

    - `n`: numero di istanze da visualizzare (default 5).

- `report_modello(modello, dati_test, target_names=None, nome_file='valutazione_modello.txt')`
Valuta un modello di classificazione su dati di test e salva il report in un file `.txt`. **Parametri:**

    - `modello`: modello addestrato.

    - `dati_test`: tuple (X_test, y_test).

    - `target_names`: nomi delle classi (opzionale).

    - `nome_file`: nome file per il report (default 'valutazione_modello.txt').

- `get_pivots_and_medoids(pivot_tree)`
Estrae i pivot discriminativi e descrittivi da ogni nodo di un PivotTree. **Parametri:**
    - `pivot_tree`: modello PivotTree addestrato.

    Ritorna: Dizionario con informazioni sui pivot per ogni nodo.

- `visualize_tree_with_pivots(pivot_tree, feature_names=None, output_file="albero_con_pivot.txt")`
Salva su file la struttura dell’albero e i dettagli sui pivot associati. **Parametri:**
    - `pivot_tree`: modello PivotTree.
    - `feature_names`: nomi delle feature (opzionale).- `output_file`: nome del file di output (default "albero_con_pivot.txt").

- `show_decision_path(pivot_tree, sample_idx, X_test, y_test)` Mostra il percorso decisionale seguito dall’albero per classificare un campione specifico. **Parametri:**
    - `pivot_tree`: modello PivotTree addestrato.
    - `sample_idx`: indice del campione da analizzare.
    - `X_test`: dati di test.
    - `y_test`: etichette reali.


## Esempio Completo
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pivot_tree_module import PivotTree 

# Caricamento dati
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Inizializzazione e addestramento del modello PivotTree
pt = PivotTree(
    max_depth=3,
    min_samples_leaf=2,
    model_type='clf',
    pairwise_metric='euclidean',
    verbose=True
)
pt.fit(X_train, y_train)

# Visualizzo le prime 5 istanze di test
print_instances_table(X_test, feature_names, n=5)

# Valuto il modello e salvo il report su file
report_modello(pt, (X_test, y_test), target_names=iris.target_names)

# Salvo la visualizzazione della struttura dell’albero con i pivot
visualize_tree_with_pivots(pt, feature_names=feature_names, output_file="albero_con_pivot.txt")

# Mostro il percorso decisionale per il primo campione di test
show_decision_path(pt, sample_idx=0, X_test=X_test, y_test=y_test)

```