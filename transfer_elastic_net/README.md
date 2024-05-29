# transfer_elastic_net

This directory contains the implementation of the [**Transfer Elastic Net**](https://doi.org/10.1101/2024.05.19.594899), a parameter transfer learning approach using $\ell_1$ and $\ell_2$ regularization terms.
The `transfer_elastic_net.py` script includes `estimateTransferElasticNet` function for performing Transfer Elastic Net via coordinate descent algorithm and the main block for examples of implementation using test dataset.


# Description of `estimateTransferElasticNet` Function

## Parameters

- **X (numpy array, shape = (n_samples, n_features)):** The data matrix where each row represents a sample and each column represents an extplanatory variable.
- **y (numpy array, shape = (n_samples,)):** The target vector where each element is the response of the corresponding row of **X**.
- **lambda_ (float):** The regularization intensity (**lamblda >= 0**). Larger values result in stronger regularization.
- **alpha (float):** The parameter controlling the balance of shrikage of estimates to 0 and **beta_tilde** (**0 ≤ alpha ≤ 1**). **alpha=1** corresponds to the conventional Elastic Net, while **alpha=0** means the algorithm uses only the terms that induce sparsity in the changes in the target estimates from the source.
- **rho (float):** The parameter controlling the balance of $\ell_1$ and $\ell_2$ regularization (**0 ≤ rho ≤ 1**). **rho=1** corresponds to the fully $\ell_1$ regularization, while **rho=0** to the fully $\ell_2$ regularization.
- **beta_tilde (numpy array, shape = (n_features,)):** Initial parameter estimates in the source domain.
- **beta_init (numpy array, shape = (n_features,), optional, default=None):** Initial values of regression coefficients used in the optimization algorithm. If not provided, a zero vector is used.
- **skip_intercept (bool, optional, default=True):** Flag to indicate whether to regularize the intercept term. If set to `True`, the intercept term is not regularized. If `False`, the intercept term is included in the regularization terms.
- **max_iter (int, optional, default=1000):** The maximum number of iterations in the optimization algorithm.
- **tol (float, optional, default=1e-4):** The tolerance for the stopping criterion of the optimization algorithm. If the changes of the parameter estimates is less than **tol**, the algorithm will stop.

## Returns

- **beta (numpy array, shape = (n_features,)):** The estimates of regression coefficients parameter.


# Example of Implementation Using Test Dataset

## Data Generation

The `create_test_data.py` script in the `test_data` directory is designed to generate two datasets corresponding to the source and target domains.
The datasets are provided in CSV format with columns labeled ['y', 'X1', 'X2', ..., 'Xp'].
The file names are:

- Source domain: `test_source_domain.csv`
- Target domain: `test_target_domain.csv`

The execution command of the script is:

```bash
python create_test_data.py
```

### Data Generation Process

#### Source Domain
The source domain data is generated according to a linear regression model:
- **Number of features:** `p = 7`
- **Number of observations:** `n = 500`
- **True Value of Coefficients:** `beta = [0, 1, -1, 0.5, -0.5, 0, 0]`

The explanatory variable matrix $X$ is drawn from a independent standard normal distribution and then standardized.
The response variable $y$ is computed through the linear model:

$y = X \beta^{\top} + \epsilon$,

where $\epsilon$ is the noise term $\epsilon \sim \mathrm{N}(0, 1^{2})$.

#### Target Domain
The target domain data shares the feature and the generation model with the source domain but the sample size is much smaller:
- **Number of features:** `p = 7`
- **Number of observations:** `n = 10`
- **True Value of Coefficients:** `beta = [0, 1, -1, 0.5, -0.5, 0, 0]`

## Implementation

The main block of `transfer_elastic_net.py` includes the implementation with the generated test datasets.
Then, the execution command of the script is:

```bash
python transfer_elastic_net.py
```

## Output

### Initial estimate in the source domain

The initial estimate in the source domain is obtained with $(\lambda,\rho,\alpha)=(0.5,0.5,1)$, which correspond to the conventional Elastic Net.
This estimate is used in the transfer learning process.

```bash
Initial estimate in the source domain: [ 0.01690669  0.46147336 -0.52298964  0.19814112 -0.19001265  0.
  0.        ]
```

### Elastic Net estimate in the target domain

The Elastic Net estimate in the target domain is obtained with $(\lambda,\rho,\alpha)=(1,0.5,1)$.

```bash
Elastic Net estimate in the target domain: [-0.47633432  0.         -0.35490085  0.1579884  -0.11971403  0.
  0.04302741]
```

### Transfer Elastic Net estimate in the target domain

The transfer Elastic Net estimate in the target domain is obtained with $(\lambda,\rho,\alpha)=(1,0.5,0.5)$.
This result reflects the information from the initial source estimate and similar to it.

```bash
Transfer Elastic Net estimate in the target domain: [-0.33979266  0.24550914 -0.52944967  0.17937543 -0.14250949  0.
  0.        ]
```

### Only transfer regularization term estimate in the target domain

The estimate using only transfer regularization term in the target domain is obtained with $(\lambda,\rho,\alpha)=(1,0.5,0)$.
This result is more similar to the initial source estimate, with the same values in the 2nd, 4th, 5th, 6th, and 7th elements.

```bash
Only transfer regularization term estimate in the target domain: [-0.22924641  0.46147336 -0.68692759  0.19814112 -0.19001265  0.
  0.        ]
```

### Lasso estimate in the target domain

The Lasso estimate in the target domain is obtained with $(\lambda,\rho,\alpha)=(1,1,1)$.

```bash
Lasso estimate in the target domain: [-0.42913184  0.         -0.36389008  0.          0.          0.
  0.        ]
```

### Transfer Lasso estimate in the target domain

The Transfer Lasso estimate in the target domain is obtained with $(\lambda,\rho,\alpha)=(1,1,0.5)$.

```bash
Transfer Lasso estimate in the target domain: [-0.32772041  0.29049459 -0.52298964  0.19814112 -0.19001265  0.
  0.        ]
```


## License

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](LICENSE).


## Credits

- **Yui Tomo**
- **Ryo Nakaki**

© Yui Tomo, Ryo Nakaki (2024).
