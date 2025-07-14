import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

def get_embeddings(data_dir, train_texts, test_texts):
    embeddings_file = os.path.join(data_dir, "embeddings_NewDataset.joblib")
    
    if os.path.exists(embeddings_file):
        print("Caricamento embedding da cache...")
        return joblib.load(embeddings_file)
        
    print("Download del modello SBERT...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Generazione embedding...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_embeddings  = model.encode(test_texts,  show_progress_bar=True)
    
    embeddings = {'train': train_embeddings, 'test': test_embeddings}
    joblib.dump(embeddings, embeddings_file)
    return embeddings

def main():

    df = pd.read_csv('new_Covid.csv')
    X_texts = df['clean_text'].astype(str).tolist()
    y       = df['classificazione']
    
 
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, random_state=42, stratify=y
    )
    

    data_dir = '.'  # cartella dove salvare il cache
    embeddings = get_embeddings(data_dir, X_train_txt, X_test_txt)
    X_train = embeddings['train']
    X_test  = embeddings['test']
    

    clf = DecisionTreeClassifier(random_state=42)
    

    param_grid = {
        'criterion': ['entropy'],
        'max_depth': [10, 5],
        'min_samples_split': [5],
        'min_samples_leaf' : [20]
    }
    

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    

    print("Migliori iperparametri:", grid.best_params_)
    best_clf = grid.best_estimator_
    

    y_pred = best_clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
