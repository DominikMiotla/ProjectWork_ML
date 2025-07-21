# PROJECT WORK ML
Questo repository raccoglie diverse implementazioni e sperimentazioni di Pivot Tree e altri modelli su differenti dataset. La struttura del repository è organizzata per facilitare la visualizzazione e la valutazione dei risultati ottenuti.

# Struttura del Repository
- `PivotTree/`
Contiene i file necessari per addestrare un Pivot Tree su un nuovo dataset, incluse le funzioni di supporto per la visualizzazione del modello.
- notebook `DrawPivotTree.ipynb` permette di graficare il Pivot Tree e salvarlo in formato `PNG`.

- `PT_Iris/` Implementazione del Pivot Tree sul dataset Iris.

- `PT_IMDB/` Implementazione del Pivot Tree sul dataset IMDB.

`PT_NoVax_G_CT/` Implementazione del Pivot Tree sul dataset NoVax, basato su testo. Utilizza feature ottenute **Group** e **Clean Text**.

- `PT_NoVax_CT/` Implementazione del Pivot Tree sul dataset NoVax, utilizzando la featur **Clean Text**.

- `PT_NoVax_CT_A/` Implementazione del Pivot Tree sul dataset NoVax, con feature **Clean Text** e **data augmentation**(dopo embedding) per la classe 0.

- `DataAugmentation_LLM/`
Contiene lo script per la data augmentation del dataset originale, focalizzata sulla classe con minore supporto, utilizzando un LLM (Grok).

- `PT_NoVax_LLM/`
Implementazione del Pivot Tree sul dataset NoVax aumentato con LLM.

- `DT_NoVax/`
Implementazione di un Decision Tree sul dataset NoVax aumentato con LLM.

- `KNN_NoVax/`
Implementazione del K-Nearest Neighbors (KNN) sul dataset NoVax aumentato con LLM.

- `Report.xlsx`
File Excel che contiene un report con i risultati ottenuti dagli esperimenti con Pivot Tree.

- `environment.yml`
File YAML contenente le dipendenze necessarie per ricreare l’ambiente Conda utilizzato nei progetti.

## Gestione dell'Ambiente Conda
### Creazione di un nuovo ambiente Conda
Per creare un nuovo ambiente Conda con tutte le dipendenze specificate nel file `environment.yml`, eseguire:

```
conda env create -f environment.yml
```
Questo comando creerà un nuovo ambiente (ad esempio, mio_ambiente) con tutte le librerie e versioni indicate.

### Aggiornamento di un ambiente Conda esistente
Per aggiornare un ambiente Conda già esistente:

```
conda env update -n mio_ambiente -f environment.yml --prune
```
L’opzione `--prune` rimuove i pacchetti non più elencati nel file YAML.