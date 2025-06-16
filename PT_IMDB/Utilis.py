from tabulate import tabulate
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_instances_table(X, feature_names, n=5):
    """
    Stampa una tabella formattata con le prime n istanze e i rispettivi valori delle feature.

    Parametri:
    - X: array o lista di liste/array, contenente i dati delle istanze
    - feature_names: lista dei nomi delle feature
    - n: numero di istanze da visualizzare (default: 5)

    Output:
    - Stampa a console una tabella con intestazioni e valori formattati con 3 decimali
    """

    headers = ["Istanza"] + feature_names
    table = []
    for i in range(min(n, len(X))):
        row = [i] + list(X[i])
        table.append(row)
    print("\n")
    print(tabulate(table, headers=headers, tablefmt="grid", floatfmt=".3f"))
    print("\n")


def report_modello(modello, dati_test, target_names=None, nome_file='valutazione_modello.txt'):
    """
    Valuta un modello di classificazione e salva i risultati in un file .txt

    Parametri:
    - modello: modello già addestrato
    - dati_test: tupla (X_test, y_test)
    - target_names: lista di nomi delle classi (opzionale)
    - nome_file: nome del file .txt (default: 'valutazione_modello.txt')
    """

    X_test, y_test = dati_test
    y_pred = modello.predict(X_test)

    #Calcolo delle metriche
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    lines = []
    lines.append("=" * 60)
    lines.append(" VALUTAZIONE DEL MODELLO ".center(60, "="))
    lines.append("=" * 60 + "\n")

    lines.append(f"Accuracy del modello: {acc:.4f}\n")

    lines.append("-" * 60)
    lines.append(" Confusion Matrix ".center(60, "-"))
    lines.append("-" * 60)


    class_labels = target_names if target_names else [str(i) for i in range(len(cm))]
    header = " " * 15 + "".join([f"{label:>12}" for label in class_labels])
    lines.append(header)
    

    for i, row in enumerate(cm):
        row_label = class_labels[i]
        row_str = f"{row_label:<15}" + "".join([f"{val:>12}" for val in row])
        lines.append(row_str)
    
    lines.append("\n" + "-" * 60)
    lines.append(" Classification Report ".center(60, "-"))
    lines.append("-" * 60)
    lines.append(report)


    with open(nome_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\t***Risultati salvati in '{nome_file}' (formato .txt)\n\n")






def get_pivots_and_medoids(pivot_tree):
    pivots_info = {}
    for node_id, node in pivot_tree._node_dict.items():
        pivots_info[node_id] = {
            'discriminative_pivots': {
                'X': node.X_pivot_discriminative,
                'y': node.y_pivot_discriminative,
                'indexes': node.discriminative_pivot_indexes,
                'names': node.discriminative_pivot_names
            },
            'descriptive_pivots': {
                'X': node.X_pivot_descriptive,
                'y': node.y_pivot_descriptive,
                'indexes': node.descriptive_pivot_indexes,
                'names': node.descriptive_pivot_names
            }
        }
    return pivots_info

def visualize_tree_with_pivots(pivot_tree, feature_names=None, output_file="albero_con_pivot.txt"):
    """
    Visualizza l'albero decisionale completo con informazioni sui pivot e medoidi,
    scrivendo il tutto su un file di testo.
    
    Parametri:
    - pivot_tree: modello PivotTree già addestrato
    - feature_names: lista dei nomi delle feature (opzionale)
    - output_file: percorso del file di testo dove scrivere l'output
    """
    
    tree_structure = pivot_tree.print_tree()
    pivots_data = get_pivots_and_medoids(pivot_tree)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(" ALBERO DECISIONALE CON PIVOT ".center(80, "=") + "\n")
        f.write("="*80 + "\n\n")
        
        f.write(tree_structure + "\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write(" DETTAGLI PIVOT PER NODO ".center(80, "-") + "\n")
        f.write("-"*80 + "\n")
        
        for node_id, node_data in pivots_data.items():
            node = pivot_tree._node_dict.get(node_id)
            if node is None:
                continue
            
            f.write(f"\nNodo {node_id}:\n")
            f.write(f"  - Tipo: {'Foglia' if node.is_leaf else 'Nodo interno'}\n")
            f.write(f"  - Classe: {node.label}\n")
            f.write(f"  - Campioni: {getattr(node, 'samples', 'N/A')}\n")
            f.write(f"  - Impurità: {getattr(node, 'impurity', 'N/A')}\n")
            
            disc_pivots = node_data.get('discriminative_pivots', {})
            disc_indexes = disc_pivots.get('indexes', [])
            if disc_indexes is not None and len(disc_indexes) > 0:
                f.write("\n  Pivot discriminativi:\n")
                for i, idx in enumerate(disc_indexes):
                    f.write(f"    - Pivot {i}: Istanza {idx}\n")
                    if feature_names is not None and 'X' in disc_pivots and i < len(disc_pivots['X']):
                        pivot_values = disc_pivots['X'][i]
                        values_str = ", ".join(f"{name}={val:.3f}" 
                                               for name, val in zip(feature_names, pivot_values))
                        f.write(f"      Valori: {values_str}\n")
            
            desc_pivots = node_data.get('descriptive_pivots', {})
            desc_indexes = desc_pivots.get('indexes', [])
            if desc_indexes is not None and len(desc_indexes) > 0:
                f.write("\n  Medoidi descrittivi:\n")
                for i, idx in enumerate(desc_indexes):
                    f.write(f"    - Medoide {i}: Istanza {idx}\n")
                    if feature_names is not None and 'X' in desc_pivots and i < len(desc_pivots['X']):
                        medoid_values = desc_pivots['X'][i]
                        values_str = ", ".join(f"{name}={val:.3f}" 
                                               for name, val in zip(feature_names, medoid_values))
                        f.write(f"      Valori: {values_str}\n")
            
            if not node.is_leaf and hasattr(node, 'pivot_used') and node.pivot_used is not None:
                f.write("\n  Pivot usati per lo split:\n")
                for pivot in (node.pivot_used if isinstance(node.pivot_used, list) else [node.pivot_used]):
                    f.write(f"    - {pivot}\n")
            
            f.write("\n" + "-"*40 + "\n")
