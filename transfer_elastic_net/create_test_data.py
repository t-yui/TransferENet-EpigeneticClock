import numpy as np
import pandas as pd

np.random.seed(100)

# setting of true parameter
beta = np.array([0, 1, -1, 0.5, -0.5, 0, 0])
p = len(beta)

# generate data for the source domain
n_source = 500
X_source = np.random.randn(n_source, p)
X_source = (X_source - np.mean(X_source, axis=0)) / np.std(X_source, axis=0)
eps_source = np.random.normal(0, 1, n_source)  # Noise term
y_source = beta @ X_source.T + eps_source
df_source = pd.DataFrame(X_source, columns=[f"X{i+1}" for i in range(p)])
df_source.insert(0, 'y', y_source)
df_source.to_csv('./test_source_domain.csv', index=False)

# generate data for the target domain
n_target = 10
X_target = np.random.randn(n_target, p)
X_target = (X_target - np.mean(X_target, axis=0)) / np.std(X_target, axis=0)
eps_target = np.random.normal(0, 1, n_target)
y_target = beta @ X_target.T + eps_target
df_target = pd.DataFrame(X_target, columns=[f"X{i+1}" for i in range(p)])
df_target.insert(0, 'y', y_target)
df_target.to_csv('./test_target_domain.csv', index=False)
