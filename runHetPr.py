#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------
# Utilities (shared)
# -----------------------------
_PTID_PAT = re.compile(r"(\d{3}_S_\d{4})")


def norm_ptid(s: str) -> Optional[str]:
    """Extract a standardized PTID like 123_S_4567 from a string; return None if not found."""
    s = str(s).strip()
    m = _PTID_PAT.search(s)
    return m.group(1) if m else None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def drop_nan_inf_cols(A: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Drop columns containing NaN/Inf. Returns (A_kept, keep_mask)."""
    A = np.asarray(A, dtype=np.float64)
    bad_nan = np.isnan(A).any(axis=0)
    bad_inf = ~np.isfinite(A).all(axis=0)
    keep = ~(bad_nan | bad_inf)
    removed = int((~keep).sum())
    if removed > 0:
        print(f"[CLEAN] {name}: removed {removed} columns with NaN/Inf; kept {int(keep.sum())}.")
    else:
        print(f"[CLEAN] {name}: no NaN/Inf columns; kept {A.shape[1]}.")
    return A[:, keep], keep


def to_chr_label(val) -> str:
    s = str(val)
    if s in ["23", "X", "x"]:
        return "chrX"
    if s in ["24", "Y", "y"]:
        return "chrY"
    if s in ["25", "MT", "Mt", "mt", "M", "m"]:
        return "chrM"
    if s.lower().startswith("chr"):
        return s if s.startswith("chr") else "chr" + s.split("chr", 1)[-1]
    return "chr" + s


def ensure_gtf_plain(gtf_path: Path, out_plain: Optional[Path] = None) -> Path:
    """If gtf is gzipped, decompress to a plain .gtf and return its path."""
    if gtf_path.suffix != ".gz":
        return gtf_path
    if out_plain is None:
        out_plain = gtf_path.with_suffix("")  # drop .gz
    if not out_plain.exists():
        print(f"[INFO] Decompressing GTF: {gtf_path} -> {out_plain}")
        with gzip.open(gtf_path, "rb") as f_in, open(out_plain, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_plain


# -----------------------------
# GWAS preparation
# -----------------------------
def detect_bfile(folder: Path) -> Path:
    """Detect exactly one PLINK trio (bed+bim+fam) in folder and return its basename Path."""
    cands = []
    for bed in folder.glob("*.bed"):
        stem = bed.with_suffix("")
        if stem.with_suffix(".bim").exists() and stem.with_suffix(".fam").exists():
            cands.append(stem)
    if len(cands) != 1:
        raise RuntimeError(f"[{folder}] expected 1 PLINK trio, got {len(cands)}: {cands}")
    return cands[0]


def _split_plink(a, b, c):
    """
    pandas_plink.read_plink returns 3 objects whose types may vary by version.
    We detect which are (G, bim, fam) robustly.
    """
    objs = [a, b, c]
    dfs = [o for o in objs if hasattr(o, "columns")]
    arrs = [o for o in objs if not hasattr(o, "columns")]
    if not (len(dfs) == 2 and len(arrs) == 1):
        raise TypeError(f"Unexpected read_plink returns: {[type(o) for o in objs]}")
    def is_bim(df): return {"chrom", "pos", "snp"}.issubset(set(df.columns))
    def is_fam(df): return "iid" in df.columns
    if is_bim(dfs[0]) and is_fam(dfs[1]):  # (bim, fam)
        return arrs[0], dfs[0], dfs[1]
    if is_bim(dfs[1]) and is_fam(dfs[0]):
        return arrs[0], dfs[1], dfs[0]
    raise ValueError(f"Cannot detect BIM/FAM. DF1 cols={list(dfs[0].columns)}, DF2 cols={list(dfs[1].columns)}")


def _get_first_series(df, col: str):
    obj = df[col]
    if hasattr(obj, "iloc") and getattr(obj, "ndim", 1) == 2:  # duplicated colnames
        return obj.iloc[:, 0]
    return obj


def _make_male_from_column(df, col: str):
    import pandas as pd
    s = _get_first_series(df, col).astype(str).str.upper().str.strip()
    male = pd.Series(np.nan, index=df.index, dtype=float)
    male[s.isin(["M", "MALE", "1"])] = 1.0
    male[s.isin(["F", "FEMALE", "0", "2"])] = 0.0
    # fuzzy fallbacks
    male[s.str.contains("MALE", na=False)] = 1.0
    male[s.str.contains("FEM",  na=False)] = 0.0
    return male


def prepare_gwas_data(
    out_dir: Path,
    diag_csv: Path,
    demo_csv: Path,
    gtf_path: Path,
    plink_dirs: List[Path],
    *,
    baseline_visit: str = "bl",
    visit_col: str = "VISCODE",
    ptid_col_diag: str = "PTID",
    diagnosis_col: str = "DIAGNOSIS",
    cn_values=(1,),
    ad_values=(3,),
    y_map=None,
    impute_missing_genotypes: bool = True,
) -> Path:
    """
    Generic GWAS prep pipeline (based on your ADNI script), but parameterized.

    Outputs to out_dir:
      - X.npy, C.npy, Y.npy, covariates.npy, ids.txt
      - markers.tsv, is_exonic.npy
      - exonMarkers.txt, nonExonMarkers.txt
      - REPORT.txt

    Assumptions / conventions (override via args if needed):
      - diag_csv includes PTID, VISCODE, DIAGNOSIS
      - baseline visit is 'bl'
      - CN/AD are encoded by cn_values/ad_values within DIAGNOSIS, mapped to 0/1
      - demo_csv includes PTID and at least one of {SEX, PTGENDER, GENDER}
      - plink_dirs each contains exactly one PLINK trio (*.bed/*.bim/*.fam)
      - GTF is hg19/GRCh37 coordinates aligned with PLINK BIM positions
    """
    import pandas as pd
    import pyranges as pr
    from pandas_plink import read_plink
    from sklearn.preprocessing import normalize

    out_dir = ensure_dir(out_dir)

    # -------- 1) Build baseline Y --------
    diag = pd.read_csv(diag_csv)
    diag[visit_col] = diag[visit_col].astype(str).str.lower()
    bl = diag[(diag[visit_col] == str(baseline_visit).lower()) &
              (diag[diagnosis_col].isin(list(cn_values) + list(ad_values)))].copy()

    if y_map is None:
        # default: CN -> 0, AD -> 1
        y_map = {int(v): 0 for v in cn_values} | {int(v): 1 for v in ad_values}

    bl["Y"] = bl[diagnosis_col].map(y_map).astype(int)
    bl["PTID"] = bl[ptid_col_diag].astype(str).apply(norm_ptid)

    y_tbl = bl.dropna(subset=["PTID"])[["PTID", "Y"]].drop_duplicates(subset=["PTID"]).reset_index(drop=True)
    print(f"[Y] n={len(y_tbl)} | counts -> 0={int((y_tbl['Y']==0).sum())}  1={int((y_tbl['Y']==1).sum())}")

    # -------- 2) Read PLINK sets, take common SNPs (keep order of first) --------
    bfiles = [detect_bfile(d) for d in plink_dirs]
    print("[INFO] Detected PLINK basenames:")
    for s in bfiles:
        print("  -", s)

    common_snps = None
    bims = []
    fams = []

    for i, stem in enumerate(bfiles):
        a, b, c = read_plink(str(stem))
        G, bim, fam = _split_plink(a, b, c)

        bim = bim.copy()
        bim["chrom"] = bim["chrom"].astype(str)
        bim = bim[(bim["chrom"] != "0") & (bim["pos"].astype(int) > 0)]
        snps = bim["snp"].astype(str).tolist()

        if common_snps is None:
            common_snps = snps
        else:
            sset = set(common_snps).intersection(snps)
            common_snps = [x for x in common_snps if x in sset]  # keep order of first batch

        bims.append(bim)
        fams.append(fam)

    if not common_snps:
        raise ValueError("No common SNPs across PLINK sets. Check build/coordinates.")

    pd.Series(common_snps).to_csv(out_dir / "markers_common.txt", index=False, header=False)
    print(f"[INFO] markers_common.txt: {len(common_snps)} SNPs")

    # -------- 3) Assemble genotype matrix (samples x common_snps), stack sets vertically --------
    blocks = []
    all_ids = []

    for bim, stem in zip(bims, bfiles):
        snp2idx = {s: i for i, s in enumerate(bim["snp"].astype(str).tolist())}
        take = [snp2idx[s] for s in common_snps if s in snp2idx]
        if len(take) != len(common_snps):
            raise RuntimeError(f"[{stem.name}] missing common SNPs unexpectedly.")

        a, b, c = read_plink(str(stem))
        G, bim2, fam = _split_plink(a, b, c)

        sub = G[take, :].compute().astype(np.float32).T  # (samples x variants)

        if impute_missing_genotypes and np.isnan(sub).any():
            col_means = np.nanmean(sub, axis=0)
            inds = np.where(np.isnan(sub))
            sub[inds] = np.take(col_means, inds[1])

        ids = fam["iid"].astype(str).apply(norm_ptid).tolist()
        all_ids.extend(ids)
        blocks.append(sub)

    G_all = np.vstack(blocks)
    pd.Series(all_ids).to_csv(out_dir / "samples_raw.txt", index=False, header=False)
    print(f"[INFO] G_all shape: {G_all.shape} | samples_raw.txt n={len(all_ids)}")

    # -------- 4) Overlap with GTF to label exonic vs non-exonic --------
    gtf_plain = ensure_gtf_plain(gtf_path, out_plain=out_dir / "annotation.gtf")
    gtf = pr.read_gtf(str(gtf_plain))
    feat_col = "feature" if "feature" in gtf.df.columns else ("Feature" if "Feature" in gtf.df.columns else None)
    if feat_col is None:
        raise KeyError(f"GTF lacks 'feature' column. Got {list(gtf.df.columns)}")
    exon = gtf[gtf.df[feat_col] == "exon"]

    # coordinates from first batch, in the same order as common_snps
    bim0 = bims[0].set_index("snp").loc[common_snps].reset_index()
    df_snps = pd.DataFrame({
        "Chromosome": bim0["chrom"].apply(to_chr_label),
        "Start": bim0["pos"].astype(int),
        "End":   bim0["pos"].astype(int) + 1,
        "Name":  bim0["snp"].astype(str),
    })
    gr_snps = pr.PyRanges(df_snps)
    ovl_df = gr_snps.join(exon).df
    exon_names = set(ovl_df["Name"].astype(str))
    is_exonic = df_snps["Name"].astype(str).isin(exon_names).to_numpy()

    n_all = len(common_snps)
    n_ex = int(is_exonic.sum())
    n_non = n_all - n_ex
    print(f"[INFO] Exon tagging: total={n_all} | exonic={n_ex} ({n_ex/n_all:.2%}) | non-exonic={n_non} ({n_non/n_all:.2%})")

    np.save(out_dir / "is_exonic.npy", is_exonic.astype(bool))
    df_snps.assign(is_exonic=is_exonic).to_csv(out_dir / "markers.tsv", sep="\t", index=False)
    with open(out_dir / "exonMarkers.txt", "w") as f:
        for rs in df_snps.loc[is_exonic, "Name"]:
            f.write(str(rs) + "\n")
    with open(out_dir / "nonExonMarkers.txt", "w") as f:
        for rs in df_snps.loc[~is_exonic, "Name"]:
            f.write(str(rs) + "\n")

    # -------- 5) Split to X/C, clean NaN/Inf loci --------
    X_full = G_all[:, is_exonic]
    C_full = G_all[:, ~is_exonic]
    X_full, maskX = drop_nan_inf_cols(X_full, "X")
    C_full, maskC = drop_nan_inf_cols(C_full, "C")

    # save post-clean marker lists
    exon_markers_after = df_snps.loc[is_exonic, "Name"].astype(str).to_numpy()[maskX]
    nonexon_markers_after = df_snps.loc[~is_exonic, "Name"].astype(str).to_numpy()[maskC]
    pd.Series(exon_markers_after).to_csv(out_dir / "exonMarkers_after_clean.txt", index=False, header=False)
    pd.Series(nonexon_markers_after).to_csv(out_dir / "nonExonMarkers_after_clean.txt", index=False, header=False)

    # -------- 6) Deduplicate + align to baseline Y by PTID --------
    ids_raw = pd.Series(all_ids, name="PTID_raw").to_frame()
    ids_raw["PTID"] = ids_raw["PTID_raw"].astype(str).apply(norm_ptid)
    ids_raw = ids_raw.dropna(subset=["PTID"]).reset_index(drop=True)

    dup_mask = ids_raw.duplicated(subset=["PTID"], keep="first")
    keep_rows = (~dup_mask).to_numpy()

    ids_dedup = ids_raw.loc[~dup_mask, ["PTID"]].reset_index(drop=True)
    X_dedup = X_full[keep_rows, :]
    C_dedup = C_full[keep_rows, :]

    merged = ids_dedup.merge(y_tbl, on="PTID", how="left")
    mask_have_y = merged["Y"].notna().to_numpy()

    X = X_dedup[mask_have_y, :]
    C = C_dedup[mask_have_y, :]
    Y = merged.loc[mask_have_y, "Y"].to_numpy(dtype=int)
    IDs = merged.loc[mask_have_y, "PTID"].to_numpy()

    print(f"[ALIGN] X={X.shape}, C={C.shape}, Y={Y.shape}, IDs={IDs.shape}")

    # -------- 7) Build covariates [sex_male, gender_male] --------
    demo = pd.read_csv(demo_csv)

    # robust PTID column detection (common conventions)
    ptid_col_demo = None
    for c in ["PTID", "ptid", "Subject", "subject", "PTIDNUM", "IID", "iid"]:
        if c in demo.columns:
            ptid_col_demo = c
            break
    if ptid_col_demo is None:
        raise KeyError("No PTID-like column found in demographics CSV.")

    # choose sex/gender columns (allow duplicates)
    cands = ["SEX", "Sex", "PTGENDER", "GENDER", "Gender", "sex", "gender"]
    sex_src = None
    gender_src = None
    for c in cands:
        if c in demo.columns:
            if sex_src is None:
                sex_src = c
            elif gender_src is None and c != sex_src:
                gender_src = c
    if sex_src is None and gender_src is None:
        raise KeyError("No sex/gender columns found in demographics CSV.")
    if sex_src is None:
        sex_src = gender_src
    if gender_src is None:
        gender_src = sex_src

    print(f"[INFO] Using demo PTID col: {ptid_col_demo} | sex src: {sex_src} | gender src: {gender_src}")

    tmp = demo[[ptid_col_demo, sex_src, gender_src]].copy()
    tmp[ptid_col_demo] = tmp[ptid_col_demo].astype(str).apply(norm_ptid)
    tmp = tmp.dropna(subset=[ptid_col_demo]).drop_duplicates(subset=[ptid_col_demo]).reset_index(drop=True)

    tmp["sex_male"] = _make_male_from_column(tmp, sex_src)
    tmp["gender_male"] = _make_male_from_column(tmp, gender_src)

    cov = (pd.DataFrame({"PTID": IDs})
           .merge(tmp[[ptid_col_demo, "sex_male", "gender_male"]].rename(columns={ptid_col_demo: "PTID"}),
                  on="PTID", how="left"))

    for col in ["sex_male", "gender_male"]:
        mode = cov[col].dropna().mode()
        fill = float(mode.iloc[0]) if not mode.empty else 0.0
        cov[col] = cov[col].astype(float).fillna(fill)

    covariates = cov[["sex_male", "gender_male"]].to_numpy(dtype=float)
    covariates = normalize(covariates, axis=0)
    print(f"[COV] covariates shape: {covariates.shape}")

    # -------- 8) Save aligned outputs --------
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "C.npy", C)
    np.save(out_dir / "Y.npy", Y)
    np.save(out_dir / "covariates.npy", covariates)
    (out_dir / "ids.txt").write_text("\n".join(map(str, IDs)) + "\n", encoding="utf-8")

    with open(out_dir / "REPORT.txt", "w", encoding="utf-8") as f:
        f.write(f"X shape: {X.shape}\nC shape: {C.shape}\nY shape: {Y.shape}\n")
        f.write(f"covariates shape: {covariates.shape}\n")
        f.write(f"IDs n: {len(IDs)}\n")
        f.write(f"exonic kept: {X.shape[1]} | non-exonic kept: {C.shape[1]}\n")
        f.write(f"common SNPs before filtering: {n_all}\n")

    print("\n[DONE] Saved aligned datasets to:", out_dir)
    return out_dir


# -----------------------------
# Fit model
# -----------------------------
def _load_prepared(folder: Path, kind: str):
    """
    Load prepared data.
    kind:
      - 'npy': X.npy/C.npy/Y.npy/covariates.npy
      - 'csv': X.csv/C.csv/Y.csv/covariates.csv (ids optional)
    """
    folder = Path(folder)

    if kind == "npy":
        X = np.load(folder / "X.npy")
        C = np.load(folder / "C.npy")
        Y = np.load(folder / "Y.npy")
        cov = np.load(folder / "covariates.npy")
        ids_path = folder / "ids.txt"
        ids = None
        if ids_path.exists():
            ids = np.array([line.strip() for line in ids_path.read_text().splitlines() if line.strip()])
        return X, C, Y, cov, ids

    if kind == "csv":
        import pandas as pd
        X = pd.read_csv(folder / "X.csv").to_numpy()
        C = pd.read_csv(folder / "C.csv").to_numpy()
        Y = pd.read_csv(folder / "Y.csv").iloc[:, 0].to_numpy()
        cov = pd.read_csv(folder / "covariates.csv").to_numpy()
        ids_path = folder / "ids.txt"
        ids = None
        if ids_path.exists():
            ids = np.array([line.strip() for line in ids_path.read_text().splitlines() if line.strip()])
        return X, C, Y, cov, ids

    raise ValueError(f"Unknown data type: {kind}")


def fit_model(prepared_dir: Path, *, show_plot: bool = True, save_plot: bool = True) -> None:
    """
    Fit PersonalizedThroughMixedModel following your existing pipeline:
      - regress out covariates from Y (LinearRegression)
      - standardize residual R
      - fit PersonalizedThroughMixedModel on (X, R, C)
      - save B.npy, pvalues.npy, and plot.
    """
    from sklearn.linear_model import LinearRegression
    from matplotlib import pyplot as plt

    # Import your model
    try:
        from model.personalizedModel import PersonalizedThroughMixedModel
    except Exception as e:
        raise ImportError(
            "Cannot import model.personalizedModel.PersonalizedThroughMixedModel. "
            "Check that your package folder is named 'model' and is installed/importable."
        ) from e

    X, C, Y, covariate, ids = _load_prepared(prepared_dir, kind="npy")

    # regress out covariates
    lr = LinearRegression()
    lr.fit(covariate, Y.astype(float))
    R = Y.astype(float) - lr.predict(covariate)
    R = (R - R.mean()) / (R.std() + 1e-8)
    print(f"[INFO] R shape = {R.shape}, mean={np.mean(R):.4f}, std={np.std(R):.4f}")

    # fit model
    pl = PersonalizedThroughMixedModel()
    B, P = pl.fit(X, R, C)

    np.save(prepared_dir / "B.npy", B)
    np.save(prepared_dir / "pvalues.npy", P)
    print(f"[SAVE] B.npy -> {prepared_dir/'B.npy'} | shape={B.shape}")
    print(f"[SAVE] pvalues.npy -> {prepared_dir/'pvalues.npy'} | shape={P.shape}")
    print(f"[INFO] Used features -> X: {X.shape[1]} | C: {C.shape[1]}")

    # plot
    plt.figure(figsize=(10, 6))
    plt.imshow(P, aspect="auto")
    plt.colorbar(label="p-value")
    plt.title("Personalized p-values")
    plt.tight_layout()

    if save_plot:
        out_png = prepared_dir / "pvalues_heatmap.png"
        plt.savefig(out_png, dpi=200)
        print(f"[SAVE] heatmap -> {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runHetPr.py", description="Prepare and run Het-PR (PersonalizedThroughMixedModel).")
    sub = p.add_subparsers(dest="command", required=True)

    # prepare subcommand
    pp = sub.add_parser("prepare", help="Prepare a GWAS dataset from raw PLINK sets + phenotype + annotations.")
    pp.add_argument("--out", required=True, help="Output folder for prepared arrays (X/C/Y/covariates).")
    pp.add_argument("--diag-csv", required=True, help="Diagnosis/phenotype CSV.")
    pp.add_argument("--demo-csv", required=True, help="Demographics CSV for covariates.")
    pp.add_argument("--gtf", required=True, help="Gencode GTF (.gtf or .gtf.gz) for exon annotation.")
    pp.add_argument("--plink-dir", action="append", required=True, help="A folder containing a PLINK trio (*.bed/*.bim/*.fam). Repeatable.")
    pp.add_argument("--baseline-visit", default="bl", help="Baseline visit label (default: bl).")
    pp.add_argument("--visit-col", default="VISCODE", help="Visit column in diagnosis CSV (default: VISCODE).")
    pp.add_argument("--ptid-col", default="PTID", help="PTID column in diagnosis CSV (default: PTID).")
    pp.add_argument("--diagnosis-col", default="DIAGNOSIS", help="Diagnosis column (default: DIAGNOSIS).")
    pp.add_argument("--cn", default="1", help="CN code(s) in DIAGNOSIS, comma-separated (default: 1).")
    pp.add_argument("--ad", default="3", help="AD code(s) in DIAGNOSIS, comma-separated (default: 3).")
    pp.add_argument("--no-impute", action="store_true", help="Disable mean-imputation for missing genotypes.")

    # fit subcommand
    pf = sub.add_parser("fit", help="Fit Het-PR model on prepared arrays.")
    pf.add_argument("-t", "--type", default="npy", choices=["npy", "csv"], help="Prepared data format (default: npy).")
    pf.add_argument("-n", "--name", required=True, help="Prepared data folder (contains X/C/Y/covariates).")
    pf.add_argument("--no-show", action="store_true", help="Do not show plot window.")
    pf.add_argument("--no-save-plot", action="store_true", help="Do not save pvalues heatmap PNG.")

    # end-to-end
    pe = sub.add_parser("prepare_and_fit", help="Prepare then fit in one command.")
    pe.add_argument("--out", required=True, help="Output folder for prepared arrays (X/C/Y/covariates).")
    pe.add_argument("--diag-csv", required=True, help="Diagnosis/phenotype CSV.")
    pe.add_argument("--demo-csv", required=True, help="Demographics CSV.")
    pe.add_argument("--gtf", required=True, help="Gencode GTF (.gtf or .gtf.gz).")
    pe.add_argument("--plink-dir", action="append", required=True, help="PLINK folder(s). Repeatable.")
    pe.add_argument("--baseline-visit", default="bl")
    pe.add_argument("--visit-col", default="VISCODE")
    pe.add_argument("--ptid-col", default="PTID")
    pe.add_argument("--diagnosis-col", default="DIAGNOSIS")
    pe.add_argument("--cn", default="1")
    pe.add_argument("--ad", default="3")
    pe.add_argument("--no-impute", action="store_true")
    pe.add_argument("--no-show", action="store_true")
    pe.add_argument("--no-save-plot", action="store_true")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        out_dir = Path(args.out)
        cn_vals = tuple(int(x) for x in str(args.cn).split(",") if str(x).strip() != "")
        ad_vals = tuple(int(x) for x in str(args.ad).split(",") if str(x).strip() != "")

        prepare_gwas_data(
            out_dir=out_dir,
            diag_csv=Path(args.diag_csv),
            demo_csv=Path(args.demo_csv),
            gtf_path=Path(args.gtf),
            plink_dirs=[Path(x) for x in args.plink_dir],
            baseline_visit=args.baseline_visit,
            visit_col=args.visit_col,
            ptid_col_diag=args.ptid_col,
            diagnosis_col=args.diagnosis_col,
            cn_values=cn_vals,
            ad_values=ad_vals,
            impute_missing_genotypes=(not args.no_impute),
        )
        return 0

    if args.command == "fit":
        prepared_dir = Path(args.name)
        # If user supplied CSV, load and save as npy in-place for consistency with fit_model
        if args.type == "csv":
            X, C, Y, cov, ids = _load_prepared(prepared_dir, kind="csv")
            np.save(prepared_dir / "X.npy", X)
            np.save(prepared_dir / "C.npy", C)
            np.save(prepared_dir / "Y.npy", Y)
            np.save(prepared_dir / "covariates.npy", cov)
            if ids is not None:
                (prepared_dir / "ids.txt").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")

        fit_model(
            prepared_dir,
            show_plot=(not args.no_show),
            save_plot=(not args.no_save_plot),
        )
        return 0

    if args.command == "prepare_and_fit":
        out_dir = Path(args.out)
        cn_vals = tuple(int(x) for x in str(args.cn).split(",") if str(x).strip() != "")
        ad_vals = tuple(int(x) for x in str(args.ad).split(",") if str(x).strip() != "")

        prepare_gwas_data(
            out_dir=out_dir,
            diag_csv=Path(args.diag_csv),
            demo_csv=Path(args.demo_csv),
            gtf_path=Path(args.gtf),
            plink_dirs=[Path(x) for x in args.plink_dir],
            baseline_visit=args.baseline_visit,
            visit_col=args.visit_col,
            ptid_col_diag=args.ptid_col,
            diagnosis_col=args.diagnosis_col,
            cn_values=cn_vals,
            ad_values=ad_vals,
            impute_missing_genotypes=(not args.no_impute),
        )

        fit_model(
            out_dir,
            show_plot=(not args.no_show),
            save_plot=(not args.no_save_plot),
        )
        return 0

    raise RuntimeError("Unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
