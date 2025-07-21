import os
import pandas as pd
from io import StringIO
import numpy as np
import requests
import math
import time
import re

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

        df['classificazione'] = df['classificazione'].str.strip().str.upper().str.replace('.', '', regex=False)
        df = df[df['classificazione'].isin(['YES', 'NO'])]
        conteggio_classi = df['classificazione'].value_counts()
        f.write("\n- CLASSI DOPO CORREZIONE:\t")
        f.write(conteggio_classi.to_string())

        f.write("\n\n " + "_" * 40 + "DATASET PROCESSED" + "_" * 40 + "\n\n")
        processed_columns = ['clean_text', 'classificazione']
        df_encoded = df[processed_columns].copy()
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

def genera_frasi_groq(prompt, n_risposte=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer KEY",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "Sei un assistente che genera frasi simili per il training NLP. Le frasi devono essere realistiche, pertinenti e in italiano."
            },
            {
                "role": "user",
                "content": f"Genera {n_risposte} frasi simili a questa: '{prompt}'"
            }
        ],
        "temperature": 0.7,
        "n": 1
    }


    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            output = response.json()
            content = output['choices'][0]['message']['content'].strip()
            

            lines = [line.strip() for line in content.split("\n") if line.strip()]
            clean_lines = []
            
            for line in lines:
                cleaned = re.sub(r'^[\d\.\-\*\)\s]+', '', line).strip()

                cleaned = cleaned.strip('"').strip("'").strip()
                if cleaned:
                    clean_lines.append(cleaned)
            
            return clean_lines[:n_risposte]
        
        except requests.exceptions.RequestException as e:
            print(f"Tentativo {attempt+1}/{max_attempts} fallito: {str(e)}")
            time.sleep(2 ** attempt)
        except (KeyError, IndexError) as e:
            print(f"Errore nella risposta API: {str(e)}")
            return []
    
    print(f"Errore Groq API dopo {max_attempts} tentativi")
    return []

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    file_path = 'Dataset/post_cleaned_it_only_GEMMA3_finale.csv'

    df = preprocessingDataset(file_path)
    print("_1: Preprocessing completato!")

    class_counts = df['classificazione'].value_counts()
    n_classe_0 = class_counts.get(0, 0)
    n_classe_1 = class_counts.get(1, 0)
    

    minority_class = 0 if n_classe_0 < n_classe_1 else 1
    da_generare = abs(n_classe_0 - n_classe_1)

    print(f"\n_2: Classe 0 (NO): {n_classe_0} - Classe 1 (YES): {n_classe_1}")
    print(f"-> Classe minoritaria: {minority_class} ({'YES' if minority_class == 1 else 'NO'})")
    print(f"-> Frasi da generare: {da_generare}")


    frasi_minoritarie = df[df['classificazione'] == minority_class]['clean_text'].tolist()
    nuove_frasi = []
    
    if da_generare > 0 and frasi_minoritarie:
        n_per_frase = math.ceil(da_generare / len(frasi_minoritarie))
        print(f"-> Generazione di {n_per_frase} frasi per ognuna delle {len(frasi_minoritarie)} frasi classe {minority_class}...")

        for i, frase in enumerate(frasi_minoritarie):
            frasi_generate = genera_frasi_groq(frase, n_risposte=n_per_frase)
            if frasi_generate:
                nuove_frasi.extend(frasi_generate)
                print(f"[{i+1}/{len(frasi_minoritarie)}] +{len(frasi_generate)} frasi")
            else:
                print(f"[{i+1}/{len(frasi_minoritarie)}] Errore nella generazione")
            
            time.sleep(2.5)

            if len(nuove_frasi) >= da_generare:
                nuove_frasi = nuove_frasi[:da_generare]
                break


    output_file = f"output/frasi_classe_{minority_class}_generate.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for frase in nuove_frasi:
            f.write(frase + "\n")

    print(f"\n_3: Salvate {len(nuove_frasi)} nuove frasi in '{output_file}'")
    print(f"    Totale frasi classe {minority_class}: {len(frasi_minoritarie) + len(nuove_frasi)}")