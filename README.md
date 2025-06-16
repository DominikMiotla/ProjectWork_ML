# Pivot Tree Project

Questo progetto implementa e addestra **Pivot Trees**, una variante degli alberi decisionali che utilizza istanze rappresentative (pivot) per effettuare le suddivisioni. 

---

## Caratteristiche principali

### Implementazione dei Pivot Tree

- **Pivot Discriminativi**  
  Punti vicini ai confini decisionali che aiutano a separare le classi.
  
- **Pivot Descrittivi (Medoids)**  
  Rappresentanti centrali dei cluster di ciascuna classe.

- **Suddivisione Ibrida**  
  Combina pivot discriminativi e descrittivi per una migliore separazione delle classi.


---

## Utilità: `Utilis.py`

### Valutazione del Modello

- Report di classificazione
- Matrici di confusione
- Metriche di accuratezza

### Visualizzazione dell'Albero

- Estrazione di pivot e medoids
- Formattazione dei percorsi decisionali
- Generazione di alberi in formato leggibile

### Ispezione dei Dati

- Visualizzazione delle istanze
- Analisi di singoli campioni

---

## Differenze Chiave: Medoids vs Pivot Discriminativi

| Caratteristica         | Medoids                        | Pivot Discriminativi           |
|------------------------|--------------------------------|---------------------------------|
| **Scopo**              | Rappresentazione della classe  | Identificazione del confine     |
| **Posizione**          | Centro del cluster             | Vicino ai confini               |
| **Selezione**          | Minimizza la distanza interna  | Massimizza la separazione       |


---


## 📊 Modelli Addestrati

### 🌸 Iris Dataset
- **Tipo**: Classificazione multi-classe (3 specie di fiori)  
- **Feature**: 4 misurazioni botaniche  
- **Uso dei Pivot**:  
  - Medoids per rappresentare fiori tipici  
  - Pivot discriminativi per distinguere specie simili

### 🎬 IMDB Sentiment Analysis
- **Tipo**: Classificazione binaria (recensioni positive/negative)  
- **Feature**: Embedding testuali (TF-IDF o Word Vectors)  
- **Uso dei Pivot**:  
  - Medoids per recensioni tipiche  
  - Pivot discriminativi per distinguere sfumature di sentiment

---

## Per iniziare

### Setup dell’Ambiente

Assicurati di avere `conda` installato, quindi esegui:

```bash
conda env create -f environment.yml
conda activate PW_ML
