import math
import numpy as np
import pandas as pd
import scanpy as sc
import os
from scipy.sparse import issparse
from scipy.special import erf
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
import gc

sc.settings.n_jobs = -1  # Use all available cores
sc.settings.verbosity = 1  # Optional: control logging level


def main():
    perturb_data_dir = "data/perturbData/"
    count = 0

    prior_data = pd.read_csv("data/prior_data.csv", sep=",")
    # Grouping prior_network network
    prior_data = prior_data.groupby("tf").agg({"action": list, "target": list})
    prior_data["updown"] = prior_data["target"].apply(lambda x: len(x))

    for file in os.listdir(perturb_data_dir):
        if file.endswith(".h5ad") and file == "TianKampmann2021_CRISPRa.h5ad":
            count += 1
            print(f"Processing file number {count}: {file}")

            adata = sc.read_h5ad(perturb_data_dir + file)

            gene_exp = pd.DataFrame(
                adata.X.toarray() if issparse(adata.X) else adata.X,
                index=adata.obs.index,
                columns=adata.var.index
            ).T
            meta_data = pd.DataFrame(adata.obs)

            # Run TF analysis
            print("Performing TF analysis...")
            min_non_na_values = int(len(gene_exp.columns) * 0.05)
            gene_exp = gene_exp.mask(gene_exp == 0).dropna(thresh=min_non_na_values)
            gene_exp = (
                    (gene_exp - gene_exp.mean(axis=1, skipna=True).values.reshape(-1, 1)) /
                    gene_exp.std(axis=1, skipna=True).values.reshape(-1, 1)
            )



if __name__ == "__main__":
    main()
