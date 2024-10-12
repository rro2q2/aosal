import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import sys
import io
import itertools
import hydra
from omegaconf import DictConfig, OmegaConf

expir_models = {
    "random": "Random",
    "entropy": "Entropy",
    "bertkm": "BERT-KM",
    "badge": "BADGE",
    "cal": "CAL",
    "aosal_mah_unc": "AOSAL-MAH-UNC",
    "aosal_mah_div": "AOSAL-MAH-DIV",
    "aosal_was_unc": "AOSAL-WAS-UNC",
    "aosal_was_div": "AOSAL-WAS-DIV",
    "asoal_constant_fpr": "AOSAL-CONST-FPR",
    "aosal_no_ood": "AOSAL-NO-OOD",
    "aosal_reg": "AOSAL-REG",
    "aosal_90": "AOSAL-FPR90",
    "aosal_95": "AOSAL-FPR95",
    "aosal_97": "AOSAL-FPR97"
}

def get_main_results(path):
    # Get model directories that are not empty
    model_dirs = [md for md in os.listdir(path) if os.listdir(f"{path}/{md}")]
    results = dict()
    for model in model_dirs:
        results[model] = dict()
        model_expir_dir = f"{path}/{model}"
        model_files = os.listdir(model_expir_dir)
        for f in model_files:
            fname = f"{model_expir_dir}/{f}"
            df = pd.read_csv(fname, encoding="utf-8")
            dropped_min_columns = [c for c in df.columns if c.split('_')[-1] == 'MIN']
            dropped_max_columns = [c for c in df.columns if c.split('_')[-1] == 'MAX']
            df = df.drop(columns=dropped_min_columns+dropped_max_columns+['Step'])
            metric = df.columns[0].split('-')[-1].strip()
            if metric == "bgt_pcnt":
                df = df.astype(int)
                results[metric] = [val for val in df[df.columns[1]].values]
            else:
                df = df.astype(float)
                results[model][metric] = []
                means = df.mean(axis=1)
                stds = df.std(axis=1)
                for i in range(len(means)):
                    results[model][metric].append((means[i], stds[i]))
        results["datasets"] = path.split('/')[1]
    return results


def plot_main_results(results, x_axis, expir_name="10", tag="test_acc"):
    tags2labels = {
        "ind_test_acc": "IND Accuracy",
        "ind_test_loss": "IND Loss",
        "ood_val_acc": "OOD Val Accuracy",
        "avg_ind": "Average IND",
        "avg_ood": "Average OOD"
    }
    if tag not in tags2labels:
        assert "Invalid tag."

    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = plt.axes()
    marker = itertools.cycle(('s', 'o', 'v', '^', '>', '.', '8', 'D', 'x', 'p'))
    linestyle = itertools.cycle(('-', '--', '-.', ':', ':', '-.', '--', '-'))
    # Set x_axis
    x_axis = x_axis if tag not in {'avg_ind', 'avg_ood'} else x_axis[1:]
    for m in results:
        if m not in {"bgt_pcnt", "datasets"}:
            metrics_avg = [m[0] for m in results[m][tag]]
            metrics_std = [m[1] for m in results[m][tag]]
            if tag in {"avg_ind", "avg_ood"}:
                metrics_avg.pop(0)
                metrics_std.pop(0)
            metrics_avg = np.array(metrics_avg)
            metrics_std = np.array(metrics_std)
            ax.plot(x_axis, metrics_avg, marker=marker.__next__(), linestyle=linestyle.__next__(), label=expir_models[m])
            ax.fill_between(x_axis, metrics_avg, metrics_avg, alpha=0.2)
    ax.legend(ncol=2)
    ax.set(xlim=(x_axis[0], x_axis[-1]),
       xlabel="Budget Size (%)", ylabel=tags2labels[tag])
    xticks = x_axis
    ax.set_xticks(xticks)
    fig.savefig(fname=f"figures/{results['datasets']}/{expir_name}/{results['datasets'].lower()}_{tag}_{expir_name}_comparison.png")

@hydra.main(version_base=None, config_path="../configs", config_name="visualizations")
def main(cfg: DictConfig):
    if cfg.ablation == True and cfg.fpr == True:
        raise ValueError("Ablation and FPR experiments cannot be both selected.\n")

    if cfg.ablation == True:
        expir_base_path = f"{cfg.expir_dir}/{cfg.ind_dataset.upper()}_{cfg.ood_dataset.upper()}/ablation"
        expir_name = "ablation"
    elif cfg.fpr == True:
        expir_base_path = f"{cfg.expir_dir}/{cfg.ind_dataset.upper()}_{cfg.ood_dataset.upper()}/fpr"
        expir_name = "fpr"
    else:
        expir_base_path = f"{cfg.expir_dir}/{cfg.ind_dataset.upper()}_{cfg.ood_dataset.upper()}/{cfg.noise}"
        expir_name = cfg.noise
    # Get main results
    results = get_main_results(expir_base_path)
    print(results)
    tags = ["ind_test_acc", "ind_test_loss", "ood_val_acc", "avg_ind", "avg_ood"]
    main_x_axis = results['bgt_pcnt']
    # Plot main results
    for t in tags:
        plot_main_results(results, main_x_axis, expir_name, tag=t)
    
if __name__ == '__main__':
    main()
