# coefficients

This directory provides coefficients for principal component (PC)–based epigenetic clocks for the **Japanese population** developed using [**Transfer Elastic Net**](https://doi.org/10.3390/math12172716).
Three CSV files are provided, corresponding to the following epigenetic clocks:

- **coefs_pc_hannum.csv**: Coefficients for the PC-Hannum clock.
- **coefs_pc_horvath.csv**: Coefficients for the PC-Horvath clock.
- **coefs_pc_phenoage.csv**: Coefficients for the PC-PhenoAge clock (Levine’s clock).

> **Note:** CpG-level coefficients are not provided due to licensing restrictions. The files here contain coefficients for the PC-based models only.


# Requirements

We recommend R version >= 4.2.2.


# Usage

## 1. Preparation of data matrix

Prepare a data matrix (`datMeth`) processed according to the flow detailed in [MorganLevineLab/PC-Clocks/run_calcPCClocks.R](https://github.com/MorganLevineLab/PC-Clocks/blob/main/run_calcPCClocks.R).
The data matrix contains methylation beta values, with samples as rows and CpG sites as columns.

## 2. PC transformation

Download the necessary PC coefficients RData file following the instruction of the [MorganLevineLab PC-Clocks repository](https://github.com/MorganLevineLab/PC-Clocks).
For example, for the PC PhenoAge clock, download [CalcPCPhenoAge.RData](https://yale.box.com/s/9eudzra5s8b0ckkwh66xi6iwuva4qct1).

Then, load the RData and transform the data matrix.

```R
load("CalcPCPhenoAge.RData")
datMethPC <- sweep(as.matrix(datMeth), 2, PCPhenoAge$center) %*% PCPhenoAge$rotation
```

## 3. Calculation of predicted values

Calculate predicted values of biological age using coefficients of the clocks.
The column named `"TENet"` contains coefficients estimated using the Transfer Elastic Net.

```R
coefs <- read.csv("coefs_pc_phenoage.csv", header = TRUE)
pred.vals <- as.numeric(datMethPC %*% coefs[2:nrow(coefs),"TENet"] + coefs[1,"TENet"])
```

Transform the obtained values when you calculate the PC-Horvath clock.

```R
anti.trafo <- function(x,adult.age=20) { ifelse(x<0, (1+adult.age)*exp(x)-1, (1+adult.age)*x+adult.age) }
pred.vals.transformed <- anti.trafo(pred.vals)
```


# License

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](LICENSE).


# Credits

Tomo, Y., & Nakaki, R. (2024). Transfer Elastic Net for developing epigenetic clocks for the Japanese population. *Mathematics*, 12(17), 2716.

© Yui Tomo, Ryo Nakaki (2024).