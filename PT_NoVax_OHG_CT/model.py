import os
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib
from PivotTree import PivotTree
import numpy as np
from Utilis import report_modello

def stampa_distribuzione(y, nome):
    distribuzione = y.value_counts(normalize=True) * 100
    print(f"\nDistribuzione percentuale delle classi nel {nome.upper()}:")
    for classe, perc in distribuzione.items():
        print(f"  Classe '{classe}': {perc:.2f}%")

def preprocessingDataset(file_path):
    df = pd.read_csv(file_path)
    output_file = "output/info_dataset.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("_" * 40 + "DATASET ORIGINALE" + "_" * 40 + "\n\n")
        f.write(f"- Numero di istanze nel dataset: {len(df)}\n")
        f.write("- FEATURES: ")
        f.write(", ".join(df.columns.tolist()) + "\n")

        f.write("- NUMERO VALORI UNICI NELLA FEATURES 'group': ")
        f.write(f" {df['group'].nunique()}\n")
        f.write("- Valori della features 'group' :")
        f.write(", ".join(map(str, df['group'].unique())) + "\n\n")

        buffer = StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue() + "\n")
        conteggio_classi = df['classificazione'].value_counts()
        f.write("- Classi presenti nel dataset:\n")
        f.write(conteggio_classi.to_string() + "\n")

        #Ci degli errori nelle etichette 
        df['classificazione'] = df['classificazione'].str.strip().str.upper().str.replace('.', '', regex=False)
        df = df[df['classificazione'].isin(['YES', 'NO'])]
        conteggio_classi = df['classificazione'].value_counts()
        f.write("\n- CLASSI DOPO CORREZIONE:\t")
        f.write(conteggio_classi.to_string())



        f.write("\n\n " + "_" * 40 + "DATASET PROCESSED" + "_" * 40 + "\n\n")
        #Selezione features
        processed_columns = ['group', 'clean_text', 'classificazione']
        df_processed = df[processed_columns]

        # One-hot encoding della colonna 'group'
        df_encoded = pd.get_dummies(df_processed, columns=['group'])

        df_encoded['classificazione'] = df_encoded['classificazione'].map({"YES": 1, "NO": 0}).astype(int)
        columns = [col for col in df_encoded.columns if col != 'classificazione'] + ['classificazione']
        df_encoded = df_encoded[columns]

        f.write("- FEATURES DEL DATASET ENCODED: ")
        f.write(", ".join(df_encoded.columns.tolist()) + "\n\n")

        buffer = StringIO()
        df_encoded.info(buf=buffer)
        f.write(buffer.getvalue() + "\n")

        conteggio_classi = df_encoded['classificazione'].value_counts()
        f.write("- Classi presenti nel dataset:\n")
        f.write(conteggio_classi.to_string() + "\n")


    return df_encoded

def get_embeddings(data_dir,train_texts,test_texts):
    embeddings_file = os.path.join(data_dir, "embeddings_CT.joblib")
    
    if os.path.exists(embeddings_file):
        print("Caricamento embedding da cache...")
        return joblib.load(embeddings_file)
        
    print("Download del modello SBERT...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Generazione embedding...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    embeddings = {
        'train': train_embeddings,
        'test': test_embeddings
    }
    
    joblib.dump(embeddings, embeddings_file)
    return embeddings


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    file_path = 'Dataset/post_cleaned_it_only_GEMMA3_finale.csv'

    df = preprocessingDataset(file_path)
    print("_1: Preprocessing completato!")
    print(df.head())

    X = df.drop(columns='classificazione')
    y = df['classificazione']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #Report numero di istanze per classe
    stampa_distribuzione(y, "intero dataset")
    stampa_distribuzione(y_train, "training set")
    stampa_distribuzione(y_test, "test set")
    print("\n\n")

    embeddings = get_embeddings("output", X_train['clean_text'].tolist(), X_test['clean_text'].tolist())

    X_train_no_text = X_train.drop(columns='clean_text')
    X_test_no_text = X_test.drop(columns='clean_text')

    # Convert embeddings to DataFrames with consistent integer column names
    n_emb_features = embeddings['train'].shape[1]
    emb_cols = list(range(n_emb_features))
    
    embeddings_train_df = pd.DataFrame(embeddings['train'], columns=emb_cols, index=X_train_no_text.index)
    embeddings_test_df = pd.DataFrame(embeddings['test'], columns=emb_cols, index=X_test_no_text.index)

    X_train_final = pd.concat([X_train_no_text, embeddings_train_df], axis=1)
    X_test_final = pd.concat([X_test_no_text, embeddings_test_df], axis=1)

    # Ensure identical column order in train/test
    X_test_final = X_test_final[X_train_final.columns]

    #Convert to NumPy arrays to avoid pandas indexing issues
    X_train_array = X_train_final.values
    X_test_array = X_test_final.values


    print("Addestramento PivotTree...")

    # Intervallo dei parametri da testare
    max_depth_values = range(3, 5)
    min_samples_leaf_values = range(1, 3)

    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            print(f"Addestramento con max_depth={max_depth}, min_samples_leaf={min_samples_leaf}...")

            # Crea e addestra il modello
            pt = PivotTree(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                model_type='clf',
                pairwise_metric='euclidean',
                random_state=42,
                allow_oblique_splits=True
            )
            pt.fit(X_train_array, y_train.values)
            dati_test = (X_test_array, y_test.values)
            nome_file = f'output/Report_MD{max_depth}_MSL{min_samples_leaf}.txt'
            report_modello(pt, dati_test, nome_file=nome_file)