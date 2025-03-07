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

UPLOAD_DIR = "data/"


def distribution_worker(max_target: int, ranks: np.array) -> np.ndarray:
    """Generate distribution for standard deviation calculation."""
    arr = np.zeros(max_target)
    cs = np.random.choice(ranks, max_target, replace=False).cumsum()
    for i in range(0, max_target):
        amr = cs[i] / (i + 1)
        arr[i] = np.min([amr, 1 - amr])
    return arr


def get_sd(max_target: int, total_genes: int, iters: int) -> np.ndarray:
    """Load or generate standard deviation distribution."""
    sd_file = f"SD_ana_{max_target}_{total_genes}_{iters}.npz"
    sd_file = os.path.join(UPLOAD_DIR, sd_file)

    if os.path.isfile(sd_file):
        print("Distribution file exists. Loading existing distribution file...")
        return np.load(sd_file)["distribution"]

    print("Distribution file does not exist. Generating new distribution file...")
    ranks = np.linspace(start=1, stop=total_genes, num=total_genes)
    ranks = (ranks - 0.5) / total_genes

    dist = Parallel(n_jobs=-1, verbose=8, backend="multiprocessing")(
        delayed(distribution_worker)(max_target, ranks) for _ in range(iters)
    )
    dist = np.std(np.array(dist).T, axis=1)
    np.savez_compressed(file=sd_file, distribution=dist)
    print("Distribution file generated successfully.")
    return dist


def sample_worker(
        sample: pd.DataFrame,
        prior_network: pd.DataFrame,
        distribution: np.array
):
    """Compute p-values for TF activity in a single sample."""
    sample.dropna(inplace=True)
    sample["rank"] = sample.rank(ascending=False)
    sample["rank"] = (sample["rank"] - 0.5) / len(sample)
    sample["rev_rank"] = 1 - sample["rank"]

    # Get target genes rank
    for tf_id, tf_row in prior_network.iterrows():
        targets = tf_row["target"]
        actions = tf_row["action"]

        # Valid target counts and total targets should be greater than 3
        valid_targets = len([t for t in targets if t in sample.index])
        if len(targets) < 3 or valid_targets < 3:
            prior_network.loc[tf_id, "rs"] = np.nan
            prior_network.loc[tf_id, "valid_target"] = np.nan
            continue

        acti_rs = 0
        inhi_rs = 0

        for i, action in enumerate(actions):
            if targets[i] in sample.index:
                if action == 1:
                    acti_rs += np.average(sample.loc[targets[i], "rank"])
                    inhi_rs += np.average(sample.loc[targets[i], "rev_rank"])
                else:
                    inhi_rs += np.average(sample.loc[targets[i], "rank"])
                    acti_rs += np.average(sample.loc[targets[i], "rev_rank"])

        rs = np.min([acti_rs, inhi_rs])
        rs = rs / valid_targets  # Average rank-sum
        prior_network.loc[tf_id, "rs"] = rs if acti_rs < inhi_rs else -rs
        prior_network.loc[tf_id, "valid_target"] = valid_targets

    # Identify non-NaN indices for 'rs' to filter the relevant rows
    valid_indices = ~np.isnan(prior_network["rs"])

    z_vals = (np.abs(prior_network.loc[valid_indices, "rs"]) - 0.5) / distribution[
        prior_network.loc[valid_indices, "valid_target"].astype(int) - 1
        ]
    p_vals = 1 + erf(z_vals / np.sqrt(2))

    # Adjust sign based on 'rs' values
    p_vals = np.where(prior_network.loc[valid_indices, "rs"] > 0, p_vals, -p_vals)

    prior_network["p-value"] = np.nan
    prior_network.loc[valid_indices, "p-value"] = p_vals

    return prior_network["p-value"].values


def bh_frd_correction(p_value_file: str, alpha=0.05) -> pd.DataFrame:
    p_value_df = pd.read_csv(p_value_file, sep="\t", index_col=0)
    p_value_df.dropna(axis=1, how="all", inplace=True)
    # Create a dataframe of shape p_value_df with all NaN values
    df_reject = pd.DataFrame(index=p_value_df.index, columns=p_value_df.columns)

    for i in p_value_df.columns:
        tf_pval = p_value_df[i].dropna()

        reject, pvals_corrected, _, _ = multipletests(
            abs(tf_pval), alpha=alpha, method="fdr_bh"
        )

        tf_pval = pd.DataFrame(tf_pval)
        tf_pval["Reject"] = reject
        df_reject[i] = tf_pval["Reject"]

    return df_reject


def run_pipeline():
    perturb_data_dir = "data/perturbData/"
    count = 0

    prior_data = pd.read_csv("data/prior_data.csv", sep=",")
    # Grouping prior_network network
    prior_data = prior_data.groupby("tf").agg({"action": list, "target": list})
    prior_data["updown"] = prior_data["target"].apply(lambda x: len(x))

    for file in os.listdir(perturb_data_dir):
        if file.endswith(".h5ad"):
            count += 1
            print(f"Processing file number {count}: {file}")

            adata = sc.read_h5ad(perturb_data_dir + file)

            gene_exp = pd.DataFrame(
                adata.X.toarray() if issparse(adata.X) else adata.X,
                index=adata.obs.index,
                columns=adata.var.index
            ).T
            meta_data = pd.DataFrame(adata.obs)
            meta_data.to_csv(f"data/{file}_meta_data.tsv", sep="\t")

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
            del adata.var["mt"]

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

            # ---------------------------------------------------------------------

            # STEP 2: TF Analysis
            # i. Perform TF analysis
            print("Performing TF analysis...")
            min_non_na_values = int(len(gene_exp.columns) * 0.05)
            gene_exp = gene_exp.mask(gene_exp == 0).dropna(thresh=min_non_na_values)
            gene_exp = (
                    (gene_exp - gene_exp.mean(axis=1, skipna=True).values.reshape(-1, 1)) /
                    gene_exp.std(axis=1, skipna=True).values.reshape(-1, 1)
            )

            print("Getting SD distribution...")
            distr = get_sd(
                max_target=np.max(prior_data["updown"]),
                total_genes=len(gene_exp),
                iters=10000,
            )

            print("Running analysis in multiple cores...")
            gene_exp = gene_exp.T
            parallel = Parallel(n_jobs=-1, verbose=8, backend="multiprocessing")
            output = parallel(
                delayed(sample_worker)(pd.DataFrame(row), prior_data, distr)
                for idx, row in gene_exp.iterrows()
            )
            p_values = pd.DataFrame(
                output,
                columns=prior_data.index,
                index=gene_exp.index
            )
            p_values.dropna(axis=1, how="all", inplace=True)

            # Save p-values
            p_file = f"data/{file}_p_values.tsv"
            p_values.to_csv(p_file, sep="\t")

            # ii. Perform Benjamini-Hochberg correction
            print("Performing Benjamini-Hochberg correction...")
            bh_df = bh_frd_correction(p_file)
            bh_file = f"data/{file}_bh_correction.tsv"
            bh_df.to_csv(bh_file, sep="\t")

            # Clear original data from memory
            del gene_exp, output
            gc.collect()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
