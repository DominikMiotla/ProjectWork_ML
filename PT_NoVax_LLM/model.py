import os
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib
from PivotTree import PivotTree
import numpy as np
from imblearn.over_sampling import SMOTE
from Utilis import report_modello
from sklearn.preprocessing import MinMaxScaler

def stampa_distribuzione(y, nome):
    distribuzione = y.value_counts(normalize=True) * 100
    print(f"\nDistribuzione percentuale delle classi nel {nome.upper()}:")
    for classe, perc in distribuzione.items():
        print(f"  Classe '{classe}': {perc:.2f}%")

def get_embeddings(data_dir,train_texts,test_texts):
    embeddings_file = os.path.join(data_dir, "embeddings_NewDataset.joblib")
    
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
    df = pd.read_csv('Dataset/new_Covid.csv')
    X = df['clean_text']
    y = df['classificazione']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    stampa_distribuzione(y, "Dataset")
    stampa_distribuzione(y_train, "training set")
    stampa_distribuzione(y_test, "test set")
    print("\n")
    print("Shape di X_train:", X_train.shape)
    print("Shape di X_test:", X_test.shape)
    print("\n\n")

    print("2: Embeddingd")

    embeddings = get_embeddings("output", X_train.tolist(), X_test.tolist())

    train_embeddings = embeddings['train']
    test_embeddings = embeddings['test']

    print("Shape di train_embeddings:", train_embeddings.shape)
    print("Shape di test_embeddings:", test_embeddings.shape)
    

    max_depth = 2
    min_samples_leaf = 4
    pt = PivotTree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        model_type='clf',
        pairwise_metric='euclidean',
        random_state=42,
        allow_oblique_splits=True,
        force_oblique_splits = False,
    )

    pt.fit(train_embeddings, y_train.values)
    dati_test = (test_embeddings, y_test.values)
    nome_file = f'output/Report_MD_FOS{max_depth}_MSL{min_samples_leaf}.txt'
    report_modello(pt, dati_test, nome_file=nome_file)