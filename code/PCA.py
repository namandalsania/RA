import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

file_path = './data/updated_dataset.csv' 
df = pd.read_csv(file_path)

encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['amount', 'involvement', 'payment_method', 'transaction_type']])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)

pca = PCA(n_components=None)  
pca.fit(scaled_data)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)

# Calculate cumulative variance explained
cumulative_variance = explained_variance.cumsum()

# Find the number of components that explain 90% of the variance
num_components_90_variance = (cumulative_variance <= 0.90).sum() + 1
print(f"Number of components explaining 90% of the variance: {num_components_90_variance}")
