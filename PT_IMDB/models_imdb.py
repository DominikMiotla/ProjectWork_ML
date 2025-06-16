import os
import numpy as np
import joblib
from datetime import datetime
from itertools import product
from sentence_transformers import SentenceTransformer
import tensorflow_datasets as tfds

from PivotTree import PivotTree

CURRENT_DIR = os.getcwd()
TFDS_DIR = os.path.join(CURRENT_DIR, "tensorflow_datasets")
MODEL_DIR = os.path.join(CURRENT_DIR, "modelli_cache")
DATASET_CACHE = os.path.join(TFDS_DIR, "imdb_dataset.joblib")
EMBEDDINGS_CACHE = os.path.join(TFDS_DIR, "embeddings.joblib")
REPORT_FILE = os.path.join(MODEL_DIR, "pivot_tree_report.txt")

os.makedirs(TFDS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


#Caricamento del dataset con cache
def load_dataset():
    if os.path.exists(DATASET_CACHE):
        print("Caricamento dataset da cache...")
        return joblib.load(DATASET_CACHE)

    print("Download del dataset IMDB...")
    (ds_train, ds_test), _ = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        download=True,
        data_dir=TFDS_DIR
    )

    def extract_data(dataset):
        texts, labels = [], []
        for text, label in tfds.as_numpy(dataset):
            texts.append(text.decode('utf-8'))
            labels.append(label)
        return texts, np.array(labels)

    train_texts, train_labels = extract_data(ds_train)
    test_texts, test_labels = extract_data(ds_test)

    data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }

    joblib.dump(data, DATASET_CACHE)
    return data


#Generazione embedding SBERT con cache
def get_embeddings(train_texts, test_texts):
    if os.path.exists(EMBEDDINGS_CACHE):
        print("Caricamento embedding da cache...")
        return joblib.load(EMBEDDINGS_CACHE)

    print("Download del modello SBERT...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Generazione embedding...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_embeddings = model.encode(test_texts, show_progress_bar=True)

    embeddings = {
        'train': train_embeddings,
        'test': test_embeddings
    }

    joblib.dump(embeddings, EMBEDDINGS_CACHE)
    return embeddings


#Addestramento e valutazione PivotTree
def train_and_evaluate_pivot_tree(params, idx, train_embeddings, train_labels, test_embeddings, test_labels):
    param_str = "_".join([f"{name}-{val}" for name, val in zip(PARAM_NAMES, params)])
    tree_file = os.path.join(MODEL_DIR, f"pivot_tree_{param_str}.pkl")

    if os.path.exists(tree_file):
        print(f"[{idx+1}/{len(PARAM_COMBINATIONS)}] Caricamento da cache: {param_str}")
        pt = joblib.load(tree_file)
    else:
        print(f"[{idx+1}/{len(PARAM_COMBINATIONS)}] Addestramento modello: {param_str}")
        pt = PivotTree(
            max_depth=params[0],
            min_samples_leaf=params[1],
            allow_oblique_splits=params[2],
            model_type='clf',
            pairwise_metric='euclidean',
            random_state=42,
            verbose=False
        )
        pt.fit(train_embeddings, train_labels)
        joblib.dump(pt, tree_file)

    predictions = pt.predict(test_embeddings)
    accuracy = np.mean(predictions == test_labels)
    print(f"  --> Accuracy: {accuracy:.4f}")

    with open(REPORT_FILE, "a") as f:
        f.write(f"[{datetime.now()}] Params: {param_str} | Accuracy: {accuracy:.4f}\n")

    return pt, accuracy

#Parametri Pivottree
PARAM_GRID = {
    "max_depth": [3, 4],
    "min_samples_leaf": [3, 8],
    "allow_oblique_splits": [False, True]
}
PARAM_NAMES = list(PARAM_GRID.keys())
PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))


if __name__ == "__main__":
    # Caricamento dati
    dataset = load_dataset()
    train_texts = dataset['train_texts']
    train_labels = dataset['train_labels']
    test_texts = dataset['test_texts']
    test_labels = dataset['test_labels']

    # Generazione embedding
    embeddings = get_embeddings(train_texts, test_texts)
    train_embeddings = embeddings['train']
    test_embeddings = embeddings['test']

    # Reset report
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)

    for idx, param_set in enumerate(PARAM_COMBINATIONS):
        train_and_evaluate_pivot_tree(
            param_set,
            idx,
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels
        )
