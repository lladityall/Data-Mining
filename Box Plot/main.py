import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate random data for 4 categories
np.random.seed(42)
random_data = pd.DataFrame({
    'Fresh': np.random.normal(loc=8000, scale=2000, size=50),
    'Milk': np.random.normal(loc=5000, scale=1500, size=50),
    'Grocery': np.random.normal(loc=6000, scale=1800, size=50),
    'Detergents_Paper': np.random.normal(loc=2000, scale=500, size=50),
})

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=random_data)
plt.title('Box Plot of Random Wholesale Data')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
