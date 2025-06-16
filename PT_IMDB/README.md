# PivotTree per l'Analisi del Sentimento su Recensioni IMDB

Questo progetto implementa una variante degli alberi decisionali chiamata **PivotTree**, dataset di recensioni cinematografiche IMDB. Il modello PivotTree utilizza **pivot discriminanti** e **medoidi descrittivi** per creare confini decisionali interpretabili in spazi di embedding ad alta dimensione.

---

## ✨ Caratteristiche Principali

- 🚀 Albero decisionale basato su pivot che opera nello spazio degli embedding  
- 🔍 Percorsi decisionali interpretabili con visualizzazione dei pivot  
- 📊 Tuning automatico degli iperparametri per prestazioni ottimali  
- 💾 Meccanismo di caching per embedding e dataset  
- 📝 Report di valutazione completi con visualizzazioni  


## 🚀 Utilizzo

### Allenamento e Valutazione del Modello

```bash
python models_imdb.py
```

Questo script esegue:

- Download e preprocessing del dataset IMDB  
- Generazione degli embedding delle frasi usando SBERT  
- Tuning degli iperparametri  
- Addestramento del miglior modello PivotTree  
- Generazione dei report di valutazione  

### Visualizzazione dell'Albero Decisionale

```python
from Utilis import visualize_tree_with_pivots

# Dopo l'addestramento del modello
visualize_tree_with_pivots(pt, output_file="tree_visualization.txt")
```

### Analisi dei Percorsi Decisionali

```python
from Utilis import show_decision_path

# Mostra il percorso decisionale per un campione specifico del test set
show_decision_path(pt, sample_idx=2, X_test=test_embeddings, y_test=test_labels)
```

---

## 🗂️ Struttura del Progetto

```
pivottree-imdb-sentiment/
├── models_imdb.py           # Script principale per training e valutazione
├── PivotTree.py             # Implementazione di PivotTree
├── RuleTree.py              # Implementazione base RuleTree
├── Utilis.py                # Funzioni di utilità per la visualizzazione
├── tensorflow_datasets/     # Directory di cache per i dataset
├── modelli_cache/           # Directory di cache per i modelli
├── valutazione_modello.txt  # Report di valutazione del modello
├── pivot_tree_report.txt    # Risultati del tuning degli iperparametri
├── albero_con_pivot.txt     # Output della visualizzazione dell'albero
└── README.md
```

---

## 📈 Risultati

Il miglior modello raggiunge un'accuratezza del **78.11%** sul test set IMDB:

```
Accuratezza: 0.7811

Matrice di Confusione:
              0       1
          0   9319    3181
          1   2291   10209

Report di Classificazione:
              precision  recall  f1-score   support
           0       0.80    0.75      0.77     12500
           1       0.76    0.82      0.79     12500
```

---

## 🧠 Concetti Chiave

### Componenti di PivotTree

- **Pivot Discriminanti**: Istanti che separano meglio le classi  
- **Medoidi Descrittivi**: Rappresentanti centrali di ogni classe  
- **Split Obliqui**: Confini decisionali che usano combinazioni di più feature  

### Workflow del Modello

1. Generazione degli embedding delle frasi con SBERT  
2. Calcolo della matrice di distanza tra gli esempi  
3. Per ogni nodo:
   - Identificazione dei medoidi descrittivi  
   - Selezione dei pivot discriminanti  
   - Creazione di confini decisionali usando i pivot  
   - Costruzione ricorsiva dell’albero  

---
