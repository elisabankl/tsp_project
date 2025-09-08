import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import json
import os

import json
import pandas as pd

import math
from typing import Sequence

# add global font size settings for plots
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 16
TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 16


def _safe_load_json(path):
    """Load JSON but ignore leading // comments (some files include a filepath comment)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("//")]
    return json.loads("\n".join(lines))

def load_instance_results(results_root="results2", instance_type=None, required_solvers=None):
    """
    Read per-instance JSON files and return three DataFrames:
      - summary_df: one row per (instance, solver) with deterministic reward, nn_choice_percentage, tour, metadata
      - nondet_df: rows (instance, solver, run_idx, reward, nn_percent)
      - perm_df: rows (instance, solver, perm_idx, reward, nn_percent)
    Only instances that contain all solvers in `required_solvers` (or in the intersection if None) are kept.
    """
    # collect all json files under results_root (optionally filtered by instance_type)
    search_root = os.path.join(results_root, instance_type) if instance_type else results_root
    json_files = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if fn.endswith(".json"):
                json_files.append(os.path.join(root, fn))


    instances = []
    solver_sets = []
    for fp in json_files:
        try:
            data = _safe_load_json(fp)
        except Exception:
            continue
        solvers = set(data.get("solvers", {}).keys())
        instances.append((fp, data))
        solver_sets.append(solvers)

    if not instances:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # determine required_solvers: use provided list or intersection of solvers present in all instances
    common = set.intersection(*solver_sets)

    
    if required_solvers is None:
        common = set.intersection(*solver_sets)
        required_solvers = sorted(common)
    else:
        required_solvers = sorted(required_solvers)

    # filter instances that have all required solvers
    filtered = [(fp, data) for fp, data in instances if set(required_solvers).issubset(set(data.get("solvers", {}).keys()))]
    summary_rows = []
    nondet_rows = []
    perm_rows = []

    for fp, data in filtered:
        meta = {
            "instance_file": fp,
            "instance_id": data.get("instance_id"),
            "p_value": data.get("p_value"),
            "instance_index": data.get("instance_index"),
            "graph_size": data.get("graph_size"),
            "instance_type": data.get("instance_type"),
            "timestamp": data.get("timestamp"),
        }
        solvers = data.get("solvers", {})
        for solver in required_solvers:
            sdat = solvers.get(solver, {})
            # deterministic summary
            summary_rows.append({
                **meta,
                "solver": solver,
                "reward": sdat.get("reward"),
                "nn_choice_percentage": sdat.get("nn_choice_percentage"),
                "tour": sdat.get("tour")
            })
            # nondet runs (if present)
            nd = sdat.get("nondet_runs", {})
            nd_rewards = nd.get("rewards", [])
            nd_nn = nd.get("nn_percentages", [])
            for i, r in enumerate(nd_rewards):
                nondet_rows.append({
                    **meta,
                    "solver": solver,
                    "run_idx": int(i),
                    "reward": r,
                    "nn_percent": (nd_nn[i] if i < len(nd_nn) else None)
                })
            # permutations (if present)
            perm = sdat.get("permutations", {})
            perm_rewards = perm.get("rewards", [])
            perm_nn = perm.get("nn_percentages", [])
            for i, r in enumerate(perm_rewards):
                perm_rows.append({
                    **meta,
                    "solver": solver,
                    "perm_idx": int(i),
                    "reward": r,
                    "nn_percent": (perm_nn[i] if i < len(perm_nn) else None)
                })

    summary_df = pd.DataFrame.from_records(summary_rows)
    summary_df["solver"] = summary_df["solver"].replace({"NearestNeighbor": "Nearest Neighbor","AntColony":"Ant Colony System"})
    nondet_df = pd.DataFrame.from_records(nondet_rows)
    perm_df = pd.DataFrame.from_records(perm_rows)

    return summary_df, nondet_df, perm_df, required_solvers


def _wilson_ci(count: int, n: int, alpha: float = 0.05):
    """Wilson score interval for proportion (returns lower, upper in fraction)."""
    if n == 0:
        return 0.0, 1.0
    phat = count / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def visualize_summary(
    summary_df: pd.DataFrame,
    agent_name: str,
    other_solvers: Sequence[str],
    output_dir: str = "plots",
    alpha: float = 0.05
):
    """
    Create required comparison visualizations from summary_df.

    Args:
      summary_df: DataFrame produced by load_instance_results (rows per instance x solver).
      agent_name: name of the RL agent solver in 'solver' column.
      other_solvers: list of solver names to compare against (e.g. ['NearestNeighbor','OR-Tools',...]).
      output_dir: directory to save png files.
      alpha: significance level for 1-alpha CI (default 0.05 -> 95% CI).

    Assumptions:
      - summary_df columns include: instance_id, p_value, solver, reward, nn_choice_percentage.
      - Only instances that have all solvers should be present (function will filter to ensure this).
    """
    os.makedirs(output_dir, exist_ok=True)

    # display name for the agent in plots and file naming: use "RL Agent" in legends,
    # but keep agent_name for data lookup and filenames
    display_agent = "RL Agent"
    instance_types_arr = summary_df['instance_type'].dropna().unique() if 'instance_type' in summary_df.columns else []
    instance_type_str = instance_types_arr[0] if len(instance_types_arr) > 0 else "unknown"

     # Determine all p values present in the data and prepare list of p values actually processed
    p_values_all = sorted(summary_df['p_value'].dropna().unique().tolist())
    p_values = []  # will contain only p values for which we computed stats (keeps x/y lengths aligned)

    # All solvers we will consider (agent + others)
    solvers = [agent_name] + list(other_solvers)

    # Prepare per-p statistics containers
    mean_rewards = {s: [] for s in solvers}
    std_rewards = {s: [] for s in solvers}
    ns = {s: [] for s in solvers}  # sample sizes (instances)
    agent_vs_other_diff = {other: {'mean': [], 'ci_lower': [], 'ci_upper': [], 'n': []} for other in other_solvers}
    better_pct = {other: {'pct': [], 'ci_lower': [], 'ci_upper': [], 'n': []} for other in other_solvers}
    nn_choice_mean = []  # mean % greedy choices for agent
    nn_choice_ci = []    # CI for that mean (t-dist)
    reward_ratio = {other: {'ratio': [], 'ci_lower': [], 'ci_upper': []} for other in other_solvers + ["best solution"]}
    pearson_corr = {other: {'r': [], 'ci_lower': [], 'ci_upper': [], 'n': []} for other in other_solvers}

    for p in p_values_all:
        dfp = summary_df[summary_df['p_value'] == p]
        # pivot rewards: index instance_id, columns solver
        rewards_wide = dfp.pivot(index='instance_id', columns='solver', values='reward')
        nn_wide = dfp.pivot(index='instance_id', columns='solver', values='nn_choice_percentage')

        # keep only rows that have all required solver rewards
        required_cols = solvers
        if not set(required_cols).issubset(rewards_wide.columns):
            # skip this p if missing solvers entirely
            continue
        rewards_wide = rewards_wide[required_cols].dropna(how='any')
        nn_wide = nn_wide.reindex(rewards_wide.index)[required_cols]  # align instances

        # only record this p as processed if we have instances to compute stats for
        n_instances = len(rewards_wide)
        if n_instances == 0:
            continue
        p_values.append(p)
        # compute mean/std for each solver
        for s in solvers:
            vals = rewards_wide[s].astype(float).values
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            mean_rewards[s].append(mean)
            std_rewards[s].append(sd)
            ns[s].append(len(vals))

        # agent nn choice percentage mean + t-CI
        agent_nn_vals = nn_wide[agent_name].dropna().astype(float).values
        if len(agent_nn_vals) > 0:
            mean_nn = np.mean(agent_nn_vals)
            sd_nn = np.std(agent_nn_vals, ddof=1) if len(agent_nn_vals) > 1 else 0.0
            df_n = max(len(agent_nn_vals) - 1, 1)
            tcrit = stats.t.ppf(1 - alpha / 2, df=df_n)
            half = tcrit * sd_nn / math.sqrt(len(agent_nn_vals)) if len(agent_nn_vals) > 1 else 0.0
            #24, because the last node is not a choice
            nn_choice_mean.append(mean_nn*100/24)
            nn_choice_ci.append(((mean_nn - half)*100/24, (mean_nn + half)*100/24))
        else:
            nn_choice_mean.append(np.nan)
            nn_choice_ci.append((np.nan, np.nan))

        # comparisons per other solver
        rewards_wide["best solution"] = rewards_wide[solvers].max(axis=1)

        for other in other_solvers:
            a_vals = rewards_wide[agent_name].astype(float).values
            o_vals = rewards_wide[other].astype(float).values

            # paired difference and t-interval for difference
            diffs = a_vals - o_vals
            mean_diff = np.mean(diffs)
            sd_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
            df_diff = max(len(diffs) - 1, 1)
            tcrit = stats.t.ppf(1 - alpha / 2, df=df_diff) if len(diffs) > 1 else 0.0
            half_diff = tcrit * sd_diff / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
            agent_vs_other_diff[other]['mean'].append(mean_diff)
            agent_vs_other_diff[other]['ci_lower'].append(mean_diff - half_diff)
            agent_vs_other_diff[other]['ci_upper'].append(mean_diff + half_diff)
            agent_vs_other_diff[other]['n'].append(len(diffs))

            # percentage agent better than other and Wilson CI
            better_count = int(np.sum(a_vals > o_vals))
            lower, upper = _wilson_ci(better_count, len(diffs), alpha=alpha)
            better_pct[other]['pct'].append(100.0 * (better_count / len(diffs)))
            better_pct[other]['ci_lower'].append(100.0 * lower)
            better_pct[other]['ci_upper'].append(100.0 * upper)
            better_pct[other]['n'].append(len(diffs))


        # Pearson correlation + Fisher z CI
            if len(a_vals) > 3:
                r, _p = stats.pearsonr(a_vals, o_vals)
                # fisher transform
                z = np.arctanh(r)
                se = 1.0 / math.sqrt(len(a_vals) - 3)
                zcrit = stats.norm.ppf(1 - alpha / 2)
                z_low, z_up = z - zcrit * se, z + zcrit * se
                r_low, r_up = np.tanh(z_low), np.tanh(z_up)
                pearson_corr[other]['r'].append(r)
                pearson_corr[other]['ci_lower'].append(r_low)
                pearson_corr[other]['ci_upper'].append(r_up)
                pearson_corr[other]['n'].append(len(a_vals))
            else:
                # insufficient samples for fisher transform
                pearson_corr[other]['r'].append(np.nan)
                pearson_corr[other]['ci_lower'].append(np.nan)
                pearson_corr[other]['ci_upper'].append(np.nan)
                pearson_corr[other]['n'].append(len(a_vals))

            # reward ratio -> replace previous ratio-of-means approach with mean-of-per-instance-ratios
            # compute per-instance ratios a_i / o_i, exclude invalid or near-zero denominators,
            # then compute t-distribution CI for the mean of ratios
        for other in other_solvers + ["best solution"]:
            a_vals = rewards_wide[agent_name].astype(float).values
            o_vals = rewards_wide[other].astype(float).values
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios_per_instance = np.array(a_vals, dtype=float) / np.array(o_vals, dtype=float)

            # mask out invalid or infinite values and near-zero denominators
            valid_mask = np.isfinite(ratios_per_instance) & np.isfinite(o_vals) & (np.abs(o_vals) > 1e-12)
            ratios_clean = ratios_per_instance[valid_mask]
            n_ratios = len(ratios_clean)

            if n_ratios > 0:
                mean_ratio = float(np.mean(ratios_clean))
                sd_ratio = float(np.std(ratios_clean, ddof=1)) if n_ratios > 1 else 0.0
                if n_ratios > 1:
                    tcrit_ratio = stats.t.ppf(1 - alpha / 2, df=n_ratios - 1)
                    half_ratio = tcrit_ratio * sd_ratio / math.sqrt(n_ratios)
                else:
                    half_ratio = 0.0
                ratio_ci_low = mean_ratio - half_ratio
                ratio_ci_up = mean_ratio + half_ratio
            else:
                mean_ratio = np.nan
                ratio_ci_low = np.nan
                ratio_ci_up = np.nan

            reward_ratio[other]['ratio'].append(mean_ratio)
            reward_ratio[other]['ci_lower'].append(ratio_ci_low)
            reward_ratio[other]['ci_upper'].append(ratio_ci_up)

            

    # ---------- PLOTTING ----------

    # 1) Average path lengths with 95% CI (t-distribution)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'pink', 'gray', 'olive', 'brown']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'p', 'h', 'd']


    plt.figure(figsize=(10, 6))
    for i,s in enumerate(solvers):
        means = [-i for i in mean_rewards[s]]
        sds = std_rewards[s]
        ns_s = ns[s]
        cis = []
        for sd, n in zip(sds, ns_s):
            if n > 1:
                tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
                cis.append(tcrit * sd / math.sqrt(n))
            else:
                cis.append(0.0)
        label = display_agent if s == agent_name else s
        plt.errorbar(p_values, means, yerr=cis, label=label,color=colors[i], marker=markers[i], capsize=3)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Average Path Length (reward)", fontsize=LABEL_FONTSIZE)
    #plt.title(f"Average Path Lengths with {int((1-alpha)*100)}% CI", fontsize=TITLE_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_path_lengths_{instance_type_str}.png"))
    plt.close()

    # 2) Differences between agent and other solvers with 95% CI (paired t CI)
    plt.figure(figsize=(10, 6))
    for i,other in enumerate(other_solvers):
        means = agent_vs_other_diff[other]['mean']
        lowers = agent_vs_other_diff[other]['ci_lower']
        uppers = agent_vs_other_diff[other]['ci_upper']
        cis = [m - l for m, l in zip(means, lowers)]
        plt.errorbar(p_values, [-i for i in means], yerr=cis, label=f"{display_agent} - {other}",color = colors[i+1],marker=markers[i+1], capsize=3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.6)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Mean Reward Difference", fontsize=LABEL_FONTSIZE)
    #plt.title(f"{display_agent} minus other solvers (paired {int((1-alpha)*100)}% CI)", fontsize=TITLE_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"agent_differences_{instance_type_str}.png"))
    plt.close()

    # 3) Percentage of instances where agent better and agent NN-choice %
    # 3a) Proportion of instances where RL Agent produced a better solution than other solvers
    plt.figure(figsize=(10, 6))
    for i, other in enumerate(other_solvers):
        pct = better_pct[other]['pct']
        low = better_pct[other]['ci_lower']
        up = better_pct[other]['ci_upper']
        # sanitize and clamp errors to non-negative
        pct_clean = [0.0 if np.isnan(x) else float(x) for x in pct]
        low_clean = [0.0 if np.isnan(x) else float(x) for x in low]
        up_clean = [100.0 if np.isnan(x) else float(x) for x in up]
        # correct ordering: error lower = mean - ci_lower, error upper = ci_upper - mean
        err_low = [max(0.0, p - l) for p, l in zip(pct_clean, low_clean)]
        err_up = [max(0.0, u - p) for p, u in zip(pct_clean, up_clean)]
        plt.errorbar(p_values, pct_clean, yerr=[err_low, err_up],
                     label=f"% {display_agent} < {other}", color=colors[i+1], marker=markers[i+1], capsize=3)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Percentage (%)", fontsize=LABEL_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"percentages_better_{instance_type_str}.png"))
    plt.close()

    # 3b) Percentage of greedy (nearest-neighbour) choices the RL Agent made (separate figure)
    plt.figure(figsize=(10, 6))
    nn_means = nn_choice_mean
    nn_lowers = [c[0] for c in nn_choice_ci]
    nn_uppers = [c[1] for c in nn_choice_ci]
    # sanitize and compute asymmetric errors
    nn_means_clean = [0.0 if np.isnan(x) else float(x) for x in nn_means]
    nn_lowers_clean = [0.0 if np.isnan(x) else float(x) for x in nn_lowers]
    nn_uppers_clean = [100.0 if np.isnan(x) else float(x) for x in nn_uppers]
    # correct ordering: lower error = mean - lower_ci, upper error = upper_ci - mean
    nn_err_low = [max(0.0, m - l) for m, l in zip(nn_means_clean, nn_lowers_clean)]
    nn_err_up = [max(0.0, u - m) for m, u in zip(nn_means_clean, nn_uppers_clean)]
    plt.errorbar(p_values, nn_means_clean, yerr=[nn_err_low, nn_err_up],
                 label=f"{display_agent} nearest neighbour choices %", marker='*', color='black', capsize=3)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Percentage (%)", fontsize=LABEL_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"percentages_nn_choice_{instance_type_str}.png"))
    plt.close()

    # 4) Reward ratio of agent and other solvers (mean of per-instance ratios) with t-CI
    plt.figure(figsize=(10, 6))
    for i, other in enumerate(other_solvers + ["best solution"]):
        ratios = np.array(reward_ratio[other]['ratio'], dtype=float)
        low = np.array(reward_ratio[other]['ci_lower'], dtype=float)
        up = np.array(reward_ratio[other]['ci_upper'], dtype=float)

        # compute asymmetric errorbars: lower = ratio - ci_lower, upper = ci_upper - ratio
        lower_err = np.where(np.isfinite(ratios) & np.isfinite(low), ratios - low, 0.0)
        upper_err = np.where(np.isfinite(ratios) & np.isfinite(up), up - ratios, 0.0)

        # clamp negatives to zero to satisfy matplotlib
        lower_err = np.maximum(lower_err, 0.0)
        upper_err = np.maximum(upper_err, 0.0)

        # plot with asymmetric errorbars
        plt.errorbar(p_values, ratios, yerr=[lower_err, upper_err],
                     label=f"{display_agent}/{other}", color=colors[i+1], marker=markers[i+1], capsize=3)

    

    plt.axhline(1.0, color='red', linestyle='--', alpha=0.6)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Reward Ratio (mean of per-instance ratios)", fontsize=LABEL_FONTSIZE)
    #plt.title("Mean of per-instance reward ratios with t-CI", fontsize=TITLE_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    # set y-axis limits as requested
    plt.ylim(0.75, 1.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reward_ratio_{instance_type_str}.png"))
    plt.close()

    # 5) Pearson correlation with Fisher CI
    plt.figure(figsize=(10, 6))
    for i, other in enumerate(other_solvers)    :
        rs = pearson_corr[other]['r']
        lows = pearson_corr[other]['ci_lower']
        ups = pearson_corr[other]['ci_upper']
        plt.errorbar(p_values, rs, yerr=[np.array(rs) - np.array(lows), np.array(ups) - np.array(rs)], label=f"{display_agent} vs {other}",color=colors[i+1], marker=markers[i+1], capsize=3)
    plt.axhline(0.0, color='black', linestyle='--', alpha=0.6)
    plt.xlabel("p", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Pearson correlation", fontsize=LABEL_FONTSIZE)
    #plt.title("Pearson correlation and Fisher-transform CI", fontsize=TITLE_FONTSIZE)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    # set y-axis limits from 0 to 1 as requested
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pearson_corr_{instance_type_str}.png"))
    plt.close()

    # return summary stats dict if user wants to inspect programmatically
    stats_out = {
        "p_values": p_values,
        "mean_rewards": mean_rewards,
        "std_rewards": std_rewards,
        "n_per_p": ns,
        "agent_vs_other_diff": agent_vs_other_diff,
        "better_pct": better_pct,
        "nn_choice_mean": nn_choice_mean,
        "nn_choice_ci": nn_choice_ci,
        "reward_ratio": reward_ratio,
        "pearson_corr": pearson_corr
    }

    # --- write CSVs with the values used to build the plots ---
    csv_dir = os.path.join(output_dir, "data")
    os.makedirs(csv_dir, exist_ok=True)

    # 1) Average path lengths (long/tidy format)
    rows = []
    for i, p in enumerate(p_values):
        for s in solvers:
            mean = mean_rewards[s][i]
            sd = std_rewards[s][i]
            n = ns[s][i]
            ci_half = 0.0
            if n > 1:
                tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
                ci_half = tcrit * sd / math.sqrt(n)
            rows.append({"p_value": p, "solver": s, "mean": mean, "std": sd, "n": n, "ci_lower": mean - ci_half, "ci_upper": mean + ci_half})
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"avg_path_lengths_{instance_type_str}.csv"), index=False)

    # 2) Paired differences (agent vs other)
    rows = []
    for other in other_solvers:
        for i, p in enumerate(p_values):
            rows.append({
                "p_value": p,
                "other_solver": other,
                "mean_diff": agent_vs_other_diff[other]['mean'][i],
                "ci_lower": agent_vs_other_diff[other]['ci_lower'][i],
                "ci_upper": agent_vs_other_diff[other]['ci_upper'][i],
                "n": agent_vs_other_diff[other]['n'][i]
            })
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"agent_differences_{instance_type_str}.csv"), index=False)

    # 3) Percentage agent better + CI
    rows = []
    for other in other_solvers:
        for i, p in enumerate(p_values):
            rows.append({
                "p_value": p,
                "other_solver": other,
                "pct_agent_better": better_pct[other]['pct'][i],
                "ci_lower": better_pct[other]['ci_lower'][i],
                "ci_upper": better_pct[other]['ci_upper'][i],
                "n": better_pct[other]['n'][i]
            })
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"better_percentage_{instance_type_str}.csv"), index=False)

    # 3b) Agent nearest-neighbour choice percent
    rows = []
    for i, p in enumerate(p_values):
        mn = nn_choice_mean[i]
        ci = nn_choice_ci[i]
        rows.append({"p_value": p, "agent_nn_choice_mean": mn, "ci_lower": ci[0], "ci_upper": ci[1]})
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"agent_nn_choice_{instance_type_str}.csv"), index=False)

    # 4) Reward ratio (mean of per-instance ratios) with t-CI
    rows = []
    for other in list(other_solvers) + ["best solution"]:
        for i, p in enumerate(p_values):
            rows.append({
                "p_value": p,
                "other_solver": other,
                "mean_ratio": reward_ratio[other]['ratio'][i],
                "ci_lower": reward_ratio[other]['ci_lower'][i],
                "ci_upper": reward_ratio[other]['ci_upper'][i]
            })
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"reward_ratio_{instance_type_str}.csv"), index=False)

    # 5) Pearson correlation with Fisher CI
    rows = []
    for other in other_solvers:
        for i, p in enumerate(p_values):
            rows.append({
                "p_value": p,
                "other_solver": other,
                "r": pearson_corr[other]['r'][i],
                "ci_lower": pearson_corr[other]['ci_lower'][i],
                "ci_upper": pearson_corr[other]['ci_upper'][i],
                "n": pearson_corr[other]['n'][i]
            })
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"pearson_corr_{instance_type_str}.csv"), index=False)

    # --- end CSV export ---

    return stats_out
instance_types = ["StringDistance", "Random", "Euclidic", "ClusteredWithRandomAsymmetry", "FlowShop"]
finetuned_agents = ["25_stringdistance_masked_ppo_20250629_130536", "25_random_masked_ppo_20250626_100225", "25_euclidic_masked_ppo_20250623_021838","25_euclidic_masked_ppo_20250623_021838","25_flowshop_masked_ppo_20250628_130538"] 



#

# ...existing code...
def generate_summary_csv(solvers: list,p_value: float = 0.1, output_csv: str = "summary_table.csv", alpha: float = 0.05):
    """
    Create a CSV with mean ± 95% CI (t-distribution) per problem type for given solvers at a specific p.
    - summary_df: output of load_instance_results (rows per instance x solver)
    - solvers: list of solver names to include (exclude agent_name if you include it separately)
    - agent_name: RL agent solver name present in summary_df
    - p_value: p to filter (e.g. 0.1)
    - output_csv: path to write CSV
    """
    instance_types = ["FlowShop", "StringDistance", "Random", "Euclidic", "ClusteredWithRandomAsymmetry"]
    finetuned_agents = ["25_flowshop_masked_ppo_20250628_130538", "25_stringdistance_masked_ppo_20250629_130536", "25_random_masked_ppo_20250626_100225", "25_euclidic_masked_ppo_20250623_021838","25_euclidic_masked_ppo_20250623_021838"]
    summary_df = pd.DataFrame()

    for i in range(5):
        temp_df, _, _, _ = load_instance_results(results_root="results2", instance_type=instance_types[i],required_solvers = [finetuned_agents[i]] + solvers)
        summary_df = pd.concat([summary_df, temp_df], ignore_index=True)
        print(summary_df.solver.unique())
    rows = []
    summary_df.instance_type.replace({"FlowShop": "Flow Shop","StringDistance": "String Dist","Euclidic":"Euclidean","ClusteredWithRandomAsymmetry":"Clusters"}, inplace=True)
    problem_types = sorted(summary_df['instance_type'].dropna().unique())
    solver_shortname_dict = {"Nearest Neighbor": "NN","Cheapest Insertion": "CI","OR-Tools": "OR","Ant Colony System": "AC"}
    summary_df.solver = summary_df.solver.map(solver_shortname_dict).fillna("RL")
    solvers = list(pd.Series(solvers).replace({"NearestNeighbor": "Nearest Neighbor","AntColony":"Ant Colony System"}))
    required = [solver_shortname_dict[s] for s in  solvers if s in solver_shortname_dict]


    for pt in problem_types:
        dfp = summary_df[(summary_df['p_value'] == p_value) & (summary_df['instance_type'] == pt)]
        if dfp.empty:
            print("dfp is empty")
            continue
        # pivot to have solvers as columns, instances as rows
        rewards_wide = dfp.pivot(index='instance_id', columns='solver', values='reward')
        print(rewards_wide.head())
        # keep only instances that have all required solvers
        if not set(required).issubset(rewards_wide.columns):
            print("Not all required solvers are present")
            continue
        rewards_wide = rewards_wide.dropna(how='any')
        if rewards_wide.shape[0] == 0:
            print("rewards_wide is empty")
            continue

        row = {"Problem Type": f"{pt} p={p_value}"}
        for solver in list(summary_df.solver.unique()):
            print(solver)
            vals = rewards_wide[solver].astype(float).values
            n = len(vals)
            mean = -np.mean(vals) if n > 0 else np.nan
            sd = np.std(vals, ddof=1) if n > 1 else 0.0
            if n > 1:
                tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
                half = tcrit * sd / math.sqrt(n)
            else:
                half = 0.0
            row[solver] = f"{mean:.2f} ± {half:.2f}"
        rows.append(row)

    if not rows:
        raise RuntimeError("No data rows generated for given p_value and solvers.")

    out_df = pd.DataFrame(rows).set_index("Problem Type")
    out_df = out_df.reindex(["Euclidean p=0.1", "Random p=0.1","Flow Shop p=0.1", "String Dist p=0.1","Clusters p=0.1"])
    out_df = out_df.reindex(columns=["RL"] + required)
    out_df.to_csv(output_csv)
    return output_csv

for i in range(5):
    summary_df, nondet_df, perm_df, required_solvers = load_instance_results(results_root="results2", instance_type=instance_types[i],required_solvers = [finetuned_agents[i],"OR-Tools", "AntColony","NearestNeighbor","Cheapest Insertion"])
    print(summary_df.solver.unique())

    visualize_summary(
    summary_df,
    finetuned_agents[i],
   ["OR-Tools", "Ant Colony System","Nearest Neighbor","Cheapest Insertion"],
   output_dir = "C:/Users/elisa/OneDrive/Dokumente/MasterThesis/Whatididsofar/master-thesis/figures",
   alpha = 0.05
)

# Example usage (run after loading summary_df):
generate_summary_csv(["NearestNeighbor","Cheapest Insertion","OR-Tools","AntColony"], p_value=0.1, output_csv="C:/Users/elisa/OneDrive/Dokumente/MasterThesis/Whatididsofar/master-thesis/tables/summary_p0_1.csv")
# filepath: c:\Users\elisa\OneDrive\Dokumente\Masterarbeit\Reinforcementlearning\tsp_project\src\comparison_plots.py
#print(csv_path)


