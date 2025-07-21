# Data Augmentation con LLM via Groq API

## Descrizione del progetto

Questo progetto ha lo scopo di migliorare le performance di un dataset di testo tramite tecniche di **data augmentation basate su modelli linguistici di grandi dimensioni (LLM)**. In particolare, utilizza l'endpoint `https://api.groq.com/openai/v1/chat/completions` per generare nuove frasi a partire da quelle già presenti nel dataset originale.

La necessità di utilizzare un LLM è nata dal fatto che **le tecniche classiche di data augmentation applicate dopo il calcolo degli embeddings non hanno prodotto risultati soddisfacenti**. Pertanto, si è scelto di generare direttamente nuove frasi sintatticamente e semanticamente coerenti con l’originale, utilizzando un modello linguistico avanzato.

## Struttura del repository
- `dataAugmentation.py`: Script per la generazione di nuove frasi utilizzando le API di Groq
- Dataset
    - `new_Covid.csv`: Dataset generato dopo l'applicazione della data augmentation
    - `post_cleaned_it_only_GEMMA3_finale.csv`: Dataset originale, prima della data augmentation
- `generateNewDataset.py`: Script che unisce i due dataset per produrre il file finale
- output
    - `/frasi_classe_0_generate.txt`: frasi nuove generate
    - `/output/info_dataset_GND.txt`: info sul nuovo dataset

# Utilizzo

1. **Preparare la chiave API Groq**  
   Per utilizzare il sistema è necessario creare una **chiave API personale** sul sito di Groq. Inserire la chiave nel file `dataAugmentation.py` nella sezione dedicata all'autenticazione.

2. **Generare nuove frasi**  
   Eseguire `dataAugmentation.py` per generare frasi augmentate a partire dal dataset originale.

3. **Creare il nuovo dataset finale**  
   Eseguire `generateNewDataset.py` per unire i dati originali e quelli generati, producendo il file `new_Covid.csv`.

# Dataset Preprocessing & Data Augmentation con Groq API
`dataAugmentation.py`: Questo script esegue il preprocessing di un dataset testuale e bilancia le classi presenti tramite data augmentation utilizzando l'API Groq LLaMA 3 per generare frasi realistiche in italiano.
## Funzionalità
### Preprocessing del Dataset
- Caricamento da CSV
- Analisi e salvataggio delle statistiche in output/info_dataset.txt
- Pulizia e standardizzazione della colonna classificazione
- Codifica binaria della classificazione (YES → 1, NO → 0)
- Output: un DataFrame df_encoded con colonne clean_text e classificazione

# Analisi delle Classi
- Stampa della distribuzione percentuale delle classi
- Identificazione della classe minoritaria
- Generazione Frasi con Groq API
- Genera frasi realistiche e pertinenti in italiano usando LLaMA 3
- Le frasi vengono create per aumentare la classe minoritaria
- Salvataggio delle frasi generate in `output/frasi_classe_<classe>_generate.txt`

## Come Usarlo
1. Imposta la **Chiave API Groq**
Sostituisci il valore "KEY" nel campo Authorization con la tua chiave API personale:

2. Esegui lo Script
Imposta il path del file CSV nella `variabile file_path`:

```python
file_path = 'Dataset/post_cleaned_it_only_GEMMA3_finale.csv'
```
3. Poi lancia lo script:
```python
python script.py
```


# Unificazione Dataset per Classificazione Testuale
`generateNewDataset.py`: Questo script unisce un dataset CSV etichettato con un file `.txt` contenente frasi da classificare come 0, creando un nuovo dataset pulito e pronto per l'addestramento di modelli di classificazione.

## Cosa fa lo script
- Pulisce il dataset CSV (corregge le etichette YES/NO, seleziona le colonne utili).
- Converte le etichette testuali in numeriche (YES → 1, NO → 0).
- Pulisce il file `.txt` rimuovendo righe indesiderate.
- Assegna la classe 0 a tutte le frasi del `.txt`.
- Unisce i due dataset: Salva il nuovo dataset finale come `Dataset/new_Covid.csv`.

### Come usarlo
1. Prepara i file:
    -  `Dataset/post_cleaned_it_only_GEMMA3_finale.csv` (testi etichettati)
    - `output/frasi_classe_0_generate.txt` (testi da etichettare come 0)

2. Esegui lo script:
```python
python generateNewDataset.py
```
### Output:
- Dataset finale: Dataset/new_Covid.csv
- report info: output/info_dataset_GND.tx