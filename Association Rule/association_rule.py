import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt

data = pd.read_csv("Groceries_dataset.csv")
data = data.astype(str)
transactions = data.values.tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary,columns = te.columns_)

frequent_itemset = apriori(df,min_support=0.01,use_colnames=True)
frequent_itemset.sort_values(by='support',ascending=False)

rules = association_rules(frequent_itemset,metric='lift',min_threshold=0.3)
rules.sort_values(by='lift',ascending=False)

top_5_rules=rules.head(5)
print(top_5_rules)

frq = df.sum().sort_values(ascending=False).head(10)
frq.plot(kind='bar',title='Top 10 items',color='skyblue')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Check the shape of top_5_rules to see if it's empty
print(top_5_rules.shape)

# If top_5_rules is empty, print a message and investigate the filtering criteria
if top_5_rules.empty:
    print("top_5_rules is empty. Check the filtering criteria in previous steps.")
else:
    strong_item = top_5_rules.iloc[0]
    print(strong_item)