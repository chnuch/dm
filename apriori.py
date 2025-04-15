import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset and preprocess
df_raw = pd.read_csv("D:\\Downloads\\transactions.csv")
dataset = df_raw.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# Transform data into one-hot encoded format
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# Run Apriori and generate rules
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print frequent itemsets and rules
print("Frequent Itemsets:", frequent_itemsets)
print("\nStrong Association Rules:", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Network graph visualization
G = nx.Graph()
for _, rule in rules.iterrows():
    antecedents = tuple(rule['antecedents'])
    consequents = tuple(rule['consequents'])
    G.add_node(antecedents)
    G.add_node(consequents)
    G.add_edge(antecedents, consequents, weight=rule['lift'])

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.6)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Network Graph of Association Rules")
plt.axis('off')
plt.show()
