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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


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

def plot_embeddings(embeddings, labels, title="Distribuzione embeddings", save_path=None):
    print("Esecuzione t-SNE per visualizzazione...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    df_plot = pd.DataFrame()
    df_plot['x'] = embeddings_2d[:, 0]
    df_plot['y'] = embeddings_2d[:, 1]
    df_plot['class'] = labels.values

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x='x', y='y', hue='class', palette='Set1', alpha=0.6)
    plt.title(title)
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend(title="Classe")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
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

    plot_embeddings(train_embeddings, y_train, save_path="output/tsne_embeddings_train.png")

    print("Addestramento PivotTree...")

    max_depth_values = [3,4,5,6]
    min_samples_leaf_values = [10,20,30,50]

    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:
            print(f"Addestramento con max_depth={max_depth}, min_samples_leaf={min_samples_leaf}...")

            pt = PivotTree(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                model_type='clf',
                pairwise_metric='euclidean',
                random_state=42,
                allow_oblique_splits=True
            )
            pt.fit(train_embeddings, y_train.values)
            dati_test = (test_embeddings, y_test.values)
            nome_file = f'output/Report_MD{max_depth}_MSL{min_samples_leaf}.txt'
            report_modello(pt, dati_test, nome_file=nome_file)
    