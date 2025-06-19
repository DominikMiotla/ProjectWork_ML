import os
import pandas as pd
from io import StringIO

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


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    file_path = 'Dataset/post_cleaned_it_only_GEMMA3_finale.csv'

    df = preprocessingDataset(file_path)
    print("_1: Preprocessing completato!")
    print(df.head())
