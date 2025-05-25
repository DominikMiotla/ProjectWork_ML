from tabulate import tabulate
def print_instances_table(X, feature_names, n=5):
    headers = ["Istanza"] + feature_names
    table = []
    for i in range(min(n, len(X))):
        row = [i] + list(X[i])
        table.append(row)
    print("\n")
    print(tabulate(table, headers=headers, tablefmt="grid", floatfmt=".3f"))
    print("\n")