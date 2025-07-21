import os
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.utils import shuffle

def stampa_distribuzione(y, nome):
    distribuzione = y.value_counts(normalize=True) * 100
    print(f"\nDistribuzione percentuale delle classi nel {nome.upper()}:")
    for classe, perc in distribuzione.items():
        print(f"  Classe '{classe}': {perc:.2f}%")

def stampa_statistiche_classi(df, colonna_classe='classificazione'):

    conteggio = df[colonna_classe].value_counts().sort_index()
    

    percentuali = (conteggio / len(df)) * 100
    

    stats_df = pd.DataFrame({
        'Numero di esempi': conteggio,
        'Percentuale (%)': percentuali.round(2)
    })

    print(stats_df)
    return stats_df


def preprocessingDataset(file_path):
    df = pd.read_csv(file_path)
    output_file = "output/info_dataset_GND.txt"

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
        processed_columns = ['clean_text', 'classificazione']
        df_processed = df[processed_columns].copy()
        

        df_processed['classificazione'] = df_processed['classificazione'].map({"YES": 1, "NO": 0}).astype(int)
        columns = [col for col in df_processed.columns if col != 'classificazione'] + ['classificazione']
        df_encoded = df_processed[columns]

        f.write("- FEATURES DEL DATASET ENCODED: ")
        f.write(", ".join(df_encoded.columns.tolist()) + "\n\n")

        buffer = StringIO()
        df_encoded.info(buf=buffer)
        f.write(buffer.getvalue() + "\n")

        conteggio_classi = df_encoded['classificazione'].value_counts()
        f.write("- Classi presenti nel dataset:\n")
        f.write(conteggio_classi.to_string() + "\n")


    return df_encoded

import pandas as pd

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]


    df = pd.DataFrame(lines, columns=['text'])


    df_cleaned = df[~df['text'].str.contains("Ecco 5 frasi simili", case=False, na=False)]


    df_features = pd.DataFrame({
        'clean_text': df_cleaned['text'],
        'classificazione': 0 
    })

    return df_features



def unisci_e_mescola(df1, df2, output_path='dataset_unificato.csv'):

    df_unito = pd.concat([df1, df2], ignore_index=True)
    

    df_shuffled = shuffle(df_unito, random_state=42).reset_index(drop=True)
    

    df_shuffled.to_csv(output_path, index=False)
    
    print(f"Dataset unificato e mescolato salvato in: {output_path}")
    stampa_statistiche_classi(df_shuffled)
    return df_shuffled



if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    file_path = 'Dataset/post_cleaned_it_only_GEMMA3_finale.csv'

    df_s = preprocessingDataset(file_path)
    print("1:Preprocessing completato!")
    print(df_s.head())

    stampa_statistiche_classi(df_s)

    print("\n\n\n")
    df_n = process_text_file("output/frasi_classe_0_generate.txt")
    print(df_n.head())
    stampa_statistiche_classi(df_n)

    unisci_e_mescola(df_s,df_n,"Dataset/new_Covid.csv")



    

