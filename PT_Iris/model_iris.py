import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from PivotTree import PivotTree
from Utilis import print_instances_table
from Utilis import report_modello
from Utilis import get_pivots_and_medoids
from Utilis import visualize_tree_with_pivots
from Utilis import show_decision_path


#Caricamento dataset + info
data = load_iris()
X = data.data
y = data.target 
feature_names = data.feature_names


print("Dataset Info:")
print(f"\tNumero di istanze: {len(X)}")
print(f"\tFeatures: {feature_names}")

num_nulls = np.isnan(X).sum()
if num_nulls > 0:
    print(f"\nATTENZIONE: Il dataset contiene {num_nulls} valori nulli!")
else:
    print("\nIl dataset NON contiene valori nulli.")

print("\nClass Info:")
classes, counts = np.unique(y, return_counts=True)
print(f"\tNumero di classi: {len(classes)}")
print(f"\tClassi: {classes.tolist()}")
for cls, count in zip(classes, counts):
    print(f"\tClasse {cls}: {count} esempi ({count / len(y) * 100:.2f}%)")

print_instances_table(X,feature_names,5)

#Pivot Tree 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pt = PivotTree(
    max_depth=4,
    min_samples_leaf=3,
    model_type='clf',
    pairwise_metric='euclidean',
    allow_oblique_splits=True
)

print("\n--- Inizio apprendimento modello ---")
pt.fit(X_train, y_train)
print("\t***Fine apprendimento modello***\n\n")

print("---Generazione file di valutazione ---")
report_modello(pt, (X_test, y_test))
print("\t***File di valutazione del modello creato***\n\n")


print("Stampa dell'albero decisioanle basato sui pivot")
visualize_tree_with_pivots(pt, feature_names=feature_names)
print("\n\n\n")

print("Percorso decisonale per la classificazione dell'istanza di test[2]")
show_decision_path(pt,2, X_test,y_test)
