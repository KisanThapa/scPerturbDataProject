import numpy as np
import pandas as pd
import scanpy as sc
import os
from scipy.sparse import issparse
from scipy.special import erf
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
import gc  # Import garbage collection module

sc.settings.n_jobs = -1  # Use all available cores
sc.settings.verbosity = 1  # Optional: control logging level

UPLOAD_DIR = "data/"


def distribution_worker(max_target: int, ranks: np.array):
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for i in range(0, max_target):
        amr = cs[i] / (i + 1)
        arr[i] = np.min([amr, 1 - amr])
    return arr


def get_sd(max_target: int, total_genes: int, iters: int):
    sd_file = f"SD_anal_{max_target}_{total_genes}_{iters}.npz"
    sd_file = os.path.join(UPLOAD_DIR, sd_file)

    if os.path.isfile(sd_file):
        print("Distribution file exists. Now we have to read it.")
        return np.load(sd_file)["distribution"]

    print("Distribution file does not exist. Now we have to generate it.")

    ranks = np.linspace(start=1, stop=total_genes, num=total_genes)
    ranks = (ranks - 0.5) / total_genes

    dist = Parallel(n_jobs=-1, verbose=3, backend="multiprocessing")(
        delayed(distribution_worker)(max_target, ranks) for _ in range(iters)
    )
    dist = np.std(np.array(dist).T, axis=1)
    np.savez_compressed(file=sd_file, distribution=dist)
    print("Distribution file generated successfully.")

    # Force garbage collection
    gc.collect()

    return dist


def sample_worker(
        sample: pd.Series,
        prior_network: pd.DataFrame,
        distribution: np.array
):
    # Convert Series to DataFrame
    sample_df = pd.DataFrame(sample)
    sample_df.dropna(inplace=True)

    # Calculate ranks in-place without creating copies
    sample_df["rank"] = sample_df.iloc[:, 0].rank(ascending=False)
    sample_df["rank"] = (sample_df["rank"] - 0.5) / len(sample_df)
    sample_df["rev_rank"] = 1 - sample_df["rank"]

    # Create results list directly (more memory efficient than DataFrame)
    results = np.full(len(prior_network), np.nan)

    # Get target genes rank
    for i, (tf_id, tf_row) in enumerate(prior_network.iterrows()):
        targets = tf_row["target"]
        actions = tf_row["action"]

        # Valid target counts and total targets should be greater than 3
        valid_targets = sum(1 for t in targets if t in sample_df.index)
        if len(targets) < 3 or valid_targets < 3:
            continue

        acti_rs = 0
        inhi_rs = 0

        for j, action in enumerate(actions):
            if targets[j] in sample_df.index:
                target_rank = sample_df.loc[targets[j], "rank"]
                target_rev_rank = sample_df.loc[targets[j], "rev_rank"]

                if action == 1:
                    acti_rs += target_rank
                    inhi_rs += target_rev_rank
                else:
                    inhi_rs += target_rank
                    acti_rs += target_rev_rank

        rs = np.min([acti_rs, inhi_rs])
        rs = rs / valid_targets  # Average rank-sum
        rs_value = rs if acti_rs < inhi_rs else -rs

        # Calculate p-value directly
        z_val = (abs(rs_value) - 0.5) / distribution[valid_targets - 1]
        p_val = 1 + erf(z_val / np.sqrt(2))

        # Adjust sign based on 'rs' value
        results[i] = p_val if rs_value > 0 else -p_val

    return results


def bh_frd_correction(p_value_file: str, alpha=0.05) -> pd.DataFrame:
    p_value_df = pd.read_csv(p_value_file, sep="\t", index_col=0)
    p_value_df.dropna(axis=1, how="all", inplace=True)

    # Create a dataframe of shape p_value_df with all NaN values
    df_reject = pd.DataFrame(index=p_value_df.index, columns=p_value_df.columns)

    for i in p_value_df.columns:
        tf_pval = p_value_df[i].dropna()

        if len(tf_pval) > 0:
            reject, _, _, _ = multipletests(
                abs(tf_pval), alpha=alpha, method="fdr_bh"
            )

            # Only assign values, don't create intermediate DataFrame
            df_reject.loc[tf_pval.index, i] = reject

        # Clear memory in each iteration
        gc.collect()

    return df_reject


def optimized_pipeline():
    perturb_data_dir = "data/perturbData/"
    count = 0

    # Load prior network data once outside the loop
    prior_data = pd.read_csv("data/prior_data.csv", sep=",")
    prior_data = prior_data.groupby("tf").agg({"action": list, "target": list})
    prior_data["updown"] = prior_data["target"].apply(lambda x: len(x))

    # Pre-calculate the distribution once
    max_target = np.max(prior_data["updown"])
    total_genes_est = 20000  # An estimate for pre-calculation, will be refined for each file

    # Try to pre-load distribution
    try:
        distr = get_sd(max_target=max_target, total_genes=total_genes_est, iters=10000)
    except Exception as e:
        print(f"Will calculate distribution for each file: {str(e)}")
        distr = None

    for file in os.listdir(perturb_data_dir):
        if file.endswith(".h5ad"):
            count += 1
            print(f"Processing file number {count}: {file}")

            # Use context manager to ensure proper resource cleanup
            adata = sc.read_h5ad(perturb_data_dir + file)

            # STEP 1: RUN UMAP PIPELINE
            # i. Filter cells and genes
            print("Filtering data...")
            sc.pp.filter_cells(adata, min_genes=3)
            sc.pp.filter_genes(adata, min_cells=200)

            # ii. Filter mitochondrial genes based on organism
            print("Filtering mitochondrial genes...")
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
            )
            adata = adata[adata.obs.pct_counts_mt < 10, :]
            adata.var.drop(columns=["mt"], inplace=True)  # Use drop instead of del

            # iii. Normalize data
            print("Normalizing data...")
            sc.pp.normalize_total(adata, target_sum=1e4)

            # iv. Log transformation
            print("Log transforming data...")
            sc.pp.log1p(adata)

            # v. Perform PCA
            print("Running PCA...")
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, n_comps=10)

            # vi. Perform UMAP
            print("Running UMAP...")
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10, metric="cosine")
            sc.tl.umap(adata, min_dist=0.1)

            # vii. Save UMAP coordinates
            print("UMAP coordinates...")
            umap_df = adata.obsm.to_df()
            umap_df.index = adata.obs_names
            umap_df["Cluster"] = "NNN"
            umap_df.to_csv(f"data/{file}_umap.tsv", sep="\t")

            # Clear UMAP data from memory
            del umap_df
            gc.collect()

            # Extract gene expression matrix efficiently
            if issparse(adata.X):
                X_array = adata.X.toarray()
            else:
                X_array = adata.X

            gene_exp = pd.DataFrame(
                X_array,
                index=adata.obs.index,
                columns=adata.var.index
            ).T

            # Clear AnnData object from memory
            del adata, X_array
            gc.collect()

            # ---------------------------------------------------------------------

            # STEP 2: TF Analysis
            # i. Perform TF analysis
            print("Performing TF analysis...")
            min_non_na_values = int(len(gene_exp.columns) * 0.05)

            # More memory efficient masking
            gene_exp = gene_exp.mask(gene_exp == 0).dropna(thresh=min_non_na_values)

            # Calculate means and stds once to avoid redundant calculations
            gene_means = gene_exp.mean(axis=1, skipna=True).values
            gene_stds = gene_exp.std(axis=1, skipna=True).values

            # In-place normalization to avoid creating a copy
            for i in range(gene_exp.shape[0]):
                gene_exp.iloc[i] = (gene_exp.iloc[i] - gene_means[i]) / gene_stds[i]

            # Force garbage collection after normalization
            del gene_means, gene_stds
            gc.collect()

            # Get distribution if not pre-calculated
            if distr is None:
                distr = get_sd(
                    max_target=max_target,
                    total_genes=len(gene_exp),
                    iters=10000,
                )

            print("Running analysis in multiple cores...")
            gene_exp = gene_exp.T

            # Use a more memory-efficient approach: process each sample and collect results
            parallel = Parallel(n_jobs=-1, verbose=3, backend="multiprocessing")
            output = parallel(
                delayed(sample_worker)(gene_exp.iloc[i], prior_data, distr)
                for i in range(len(gene_exp))
            )

            # Convert results to DataFrame more efficiently
            p_values = pd.DataFrame(
                output,
                columns=prior_data.index,
                index=gene_exp.index
            )

            # Clear original data from memory
            del gene_exp, output
            gc.collect()

            # Clean up NaNs and save
            p_values.dropna(axis=1, how="all", inplace=True)
            p_file = f"data/{file}_p_values.tsv"
            p_values.to_csv(p_file, sep="\t")

            # Clear p_values from memory
            del p_values
            gc.collect()

            # ii. Perform Benjamini-Hochberg correction
            print("Performing Benjamini-Hochberg correction...")
            bh_df = bh_frd_correction(p_file)
            bh_file = f"data/{file}_bh_correction.tsv"
            bh_df.to_csv(bh_file, sep="\t")

            # Clear final results and force garbage collection
            del bh_df
            gc.collect()

    print("Concluded...")


if __name__ == "__main__":
    optimized_pipeline()
