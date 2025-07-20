import tensorflow_datasets as tfds
import numpy as np
import os
import joblib
from sentence_transformers import SentenceTransformer
from PivotTree import PivotTree
from itertools import product
from datetime import datetime

from Utilis import report_modello
from Utilis import visualize_tree_with_pivots
from Utilis import show_decision_path

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "tensorflow_datasets")
os.makedirs(data_dir, exist_ok=True)

def load_dataset():
    dataset_file = os.path.join(data_dir, "imdb_dataset.joblib")
    
    if os.path.exists(dataset_file):
        print("Caricamento dataset da cache...")
        data = joblib.load(dataset_file)
        return data
        
    print("Download del dataset...")
    (ds_train, ds_test), ds_info = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        download=True,
        data_dir=data_dir
    )

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for text, label in tfds.as_numpy(ds_train):
        train_texts.append(text.decode('utf-8'))
        train_labels.append(label)

    for text, label in tfds.as_numpy(ds_test):
        test_texts.append(text.decode('utf-8'))
        test_labels.append(label)

    data = {
        'train_texts': train_texts,
        'train_labels': np.array(train_labels),
        'test_texts': test_texts,
        'test_labels': np.array(test_labels)
    }
    
    joblib.dump(data, dataset_file)
    return data

dataset = load_dataset()
train_texts = dataset['train_texts']
train_labels = dataset['train_labels']
test_texts = dataset['test_texts']
test_labels = dataset['test_labels']

def get_embeddings():
    embeddings_file = os.path.join(data_dir, "embeddings.joblib")
    
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

embeddings = get_embeddings()
train_embeddings = embeddings['train']
test_embeddings = embeddings['test']
nome_cartella = "Output"
os.makedirs(nome_cartella, exist_ok=True)
print("Addestramento PivotTree...")
pt = PivotTree(
    max_depth=6,
    min_samples_leaf=3,
    model_type='clf',
    pairwise_metric='euclidean',
    allow_oblique_splits = True
)

pt.fit(train_embeddings, train_labels)

predictions = pt.predict(test_embeddings)
accuracy = np.mean(predictions == test_labels)
print(f"Test Accuracy: {accuracy:.4f}")



print("---Generazione file di valutazione ---")
report_modello(pt, (test_embeddings, test_labels),nome_file='Output/valutazione_modello.txt')
print("\t***File di valutazione del modello creato***\n\n")


print("Stampa dell'albero decisioanle basato sui pivot")
visualize_tree_with_pivots(pt, output_file="Output/albero_con_pivot.txt")
print("\n\n\n")

print("Percorso decisonale per la classificazione dell'istanza di test[2]")
show_decision_path(pt,2, test_embeddings,test_labels)