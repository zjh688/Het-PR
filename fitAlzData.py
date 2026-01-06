__author__ = 'Jiahui Zhang'

from optparse import OptionParser, OptionGroup
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Your core model
from model.personalizedModel import PersonalizedThroughMixedModel


def run_exp(data_dir: Path, out_dir: Path, save_beta: bool = True, do_plot: bool = True):
    # ---- validate paths ----
    data_dir = Path(data_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"[ERROR] data_dir does not exist: {data_dir}")

    required = ["X.npy", "Y.npy", "C.npy", "covariates.npy"]
    missing = [fn for fn in required if not (data_dir / fn).exists()]
    if missing:
        raise FileNotFoundError(f"[ERROR] Missing files in {data_dir}: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load aligned data ----
    Y = np.load(data_dir / "Y.npy")                 # (n,)
    X = np.load(data_dir / "X.npy")                 # (n, p_exon)
    C = np.load(data_dir / "C.npy")                 # (n, p_nonexon)
    covariate = np.load(data_dir / "covariates.npy")  # (n, k)

    # ---- regress covariates, get residual R, standardize ----
    lr = LinearRegression()
    lr.fit(covariate, Y.astype(float))
    R = Y.astype(float) - lr.predict(covariate)
    R = (R - R.mean()) / (R.std() + 1e-8)

    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] Shapes: Y={Y.shape}, X={X.shape}, C={C.shape}, covariates={covariate.shape}")
    print(f"[INFO] R: shape={R.shape}, mean={np.mean(R):.6f}, std={np.std(R):.6f}")

    # ---- fit Personalized Mixed Model ----
    pl = PersonalizedThroughMixedModel()
    B, P = pl.fit(X, R, C)

    # ---- save outputs ----
    # Always save P; B optional because it can be large
    np.save(out_dir / "pvalues.npy", P)
    np.save(out_dir / "residual_R.npy", R)

    if save_beta:
        np.save(out_dir / "beta.npy", B)

    print(f"[SAVE] pvalues.npy   -> {out_dir / 'pvalues.npy'}   | shape={np.shape(P)}")
    print(f"[SAVE] residual_R.npy -> {out_dir / 'residual_R.npy'} | shape={np.shape(R)}")
    if save_beta:
        print(f"[SAVE] beta.npy     -> {out_dir / 'beta.npy'}     | shape={np.shape(B)}")

    # ---- plot (same as your current plot) ----
    if do_plot:
        plt.imshow(P, aspect="auto")
        plt.colorbar(label="p-value")
        plt.title("Personalized p-values")
        plt.tight_layout()
        plt.show()

    return B, P


def main():
    usage = """usage: %prog [options]
Example:
  python runPM.py -d data/example -o results --plot
"""
    parser = OptionParser(usage=usage)

    dataGroup = OptionGroup(parser, "Data Options")
    outGroup = OptionGroup(parser, "Output Options")
    plotGroup = OptionGroup(parser, "Plot Options")

    dataGroup.add_option("-d", "--data_dir", dest="data_dir", default=None,
                         help="Directory that contains X.npy, Y.npy, C.npy, covariates.npy")
    outGroup.add_option("-o", "--out_dir", dest="out_dir", default="results",
                        help="Output directory (default: results)")
    outGroup.add_option("--no_beta", action="store_true", dest="no_beta", default=False,
                        help="Do NOT save beta.npy (useful if B is too large).")

    plotGroup.add_option("--plot", action="store_true", dest="plot", default=False,
                         help="Show heatmap plot of P.")
    plotGroup.add_option("--no_plot", action="store_true", dest="no_plot", default=False,
                         help="Disable plotting (overrides --plot).")

    parser.add_option_group(dataGroup)
    parser.add_option_group(outGroup)
    parser.add_option_group(plotGroup)

    (options, args) = parser.parse_args()

    if options.data_dir is None:
        parser.print_help()
        raise SystemExit("\n[ERROR] You must provide -d / --data_dir")

    data_dir = Path(options.data_dir)
    out_dir = Path(options.out_dir)

    do_plot = options.plot and (not options.no_plot)
    save_beta = not options.no_beta

    run_exp(data_dir=data_dir, out_dir=out_dir, save_beta=save_beta, do_plot=do_plot)


if __name__ == "__main__":
    main()
