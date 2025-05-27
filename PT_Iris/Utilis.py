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