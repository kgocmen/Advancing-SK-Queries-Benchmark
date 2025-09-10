import json
import os
from typing import Dict, List
from constants import *
from sklearn.metrics.pairwise import cosine_similarity
import ast
import argparse
from Benchmark import SCENARIOS
import matplotlib.pyplot as plt

class QueryResultsParser:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()
        self.insertion_time = self.data.get("insertion_time", 0)
        self.index_creation_time = self.data.get("index_creation", None)["time"]
        self.query_results = self.data.get("query_results", {})
        self._st_model = SEMANTIC

    def _load_json(self) -> Dict:
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"File not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_query_result_id(self, qid):       return [i["id"] for i in self.query_results.get(qid, {}).get("results", [])]
    def get_query_result_distance(self, qid): return [i["distance"] for i in self.query_results.get(qid, {}).get("results", [])]
    def get_query_result_tags(self, qid):     return [i["tags"] for i in self.query_results.get(qid, {}).get("results", [])]

    
    def get_execution_time(self, query_id: str) -> float:
        return self.query_results.get(query_id, {}).get("execution_time", 0.0)
    
    def calculate_coherence_and_time(self, k: int) -> Dict:
        """
        semantic_coherence@k: mean of a (k x k) tag-equality matrix (1 if tags equal else 0)
        spatial_std_over_mean@k: std(distance) / mean(distance) over top-k
        """
        import json, statistics

        sem_sum = 0.0; sem_n = 0
        #spa_sum = 0.0; spa_n = 0
        total_exec = 0.0; max_exec = 0.0

        for qid, entry in self.query_results.items():
            total_exec += entry.get("execution_time", 0.0)
            max_exec = max(max_exec, entry.get("execution_time", 0.0))

            # --- semantic: k x k matrix of tag equality (exact match)
            tags_list = self.get_query_result_tags(qid)
            texts = [str(ast.literal_eval(t)) if isinstance(t, str) else str(t) for t in tags_list]
            if texts:
                emb = self._st_model.encode(texts)  # (k,d)
                sim_mat = cosine_similarity(emb, emb)  # k x k, in [-1,1], diag=1
                sem = float(sim_mat.mean())            # 1.0 if all identical
                sem_sum += sem; sem_n += 1

        qcount = max(1, len(self.query_results))
        return {
            "insertion_time": round(self.insertion_time, 4),
            "index_creation_time": round(self.index_creation_time, 4),
            "average_execution_time": round(total_exec / qcount, 4),
            "max_execution_time": round(max_exec, 4),
            f"semantic_coherence@{k}": round(sem_sum / sem_n, 4) if sem_n else None,
        }

    def calculate_recall_and_time(self, k: int, ground_truth_path: str, ) -> Dict:

        ground_truth_parser = QueryResultsParser(ground_truth_path)
        ground_truth = ground_truth_parser.query_results

        recall_score = 0
        total_queries = len(ground_truth)
        total_execution_time = 0.0

        if total_queries == 0:
            raise ValueError("Ground truth query results are empty.")

        for query_id in ground_truth.keys():
            true_results = set(i["id"] for i in ground_truth[str(query_id)]["results"])
            
            predicted_results = self.get_query_result_id(query_id)
            execution_time = self.get_execution_time(query_id)

            total_execution_time += execution_time 

            top_k_pred = set(predicted_results[:k])
            if true_results:
                recall_k = len(true_results & top_k_pred) / len(true_results)  #recall
            else:
                recall_k = 1
            recall_score += recall_k

        #recall
        recall_score = {f"recall@{k}": round(recall_score / total_queries, 4)}
        

        avg_execution_time = round(total_execution_time / total_queries, 4)
        max_execution_time = max([self.get_execution_time(query_id) for query_id in ground_truth.keys()])

        return {
            "insertion_time": self.insertion_time,
            "index_creation_time": self.index_creation_time,
            "average_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            **recall_score
        }
        #return {"average execution": avg_execution_time, **recall_score}

    def print_report(self, k: int):
        results = self.calculate_coherence_and_time(k)
        #results = self.calculate_recall_and_time(ground_truth_path, k)

        print(f"\nReport for {self.json_path}")
        for key, value in results.items():
            print(f"{key}: {value}")

    def __repr__(self):
        return f"QueryResultsParser({self.json_path})"

    @staticmethod
    def produce_plots_for_group(test_files: List[str], source: str, k: int, experiment: str, scenario: str):
        """
        Build plots for a (source, k) group using multiple results JSON files (one per method).
        - Line chart: per-query execution times; each method is a line (values annotated).
        - Grouped bar chart: insertion, index creation per method (values annotated).
        - Semantic coherence per query (values annotated).
        Output directory: ./results/<experiment>/plots/
        """
        import os, json
        import numpy as np
        import matplotlib.pyplot as plt

        # --- Collect data from the available files ---
        records = []  # each: {method, insertion_time, index_creation_time, avg_exec, exec_times}
        for path in test_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # infer method name from filename: <...>/<source>_<method>_results_k<k>.json
            base = os.path.basename(path)
            name = base.replace(f"{source}_", "")
            name = name.replace(f"_results_k{k}.json", "")
            method = name

            insertion_time = float(data.get("insertion_time", 0.0))
            index_creation_time = float((data.get("index_creation") or {}).get("time", 0.0))

            # per-query execution times (sorted by numeric key if possible)
            qres = data.get("query_results", {}) or {}
            def _safe_int(x):
                try:
                    return int(x)
                except Exception:
                    return x
            keys_sorted = sorted(qres.keys(), key=_safe_int)
            exec_times = [float(qres[k]["execution_time"]) for k in keys_sorted if "execution_time" in qres.get(k, {})]

            avg_exec = float(np.mean(exec_times)) if exec_times else 0.0

            records.append({
                "method": method,
                "insertion_time": insertion_time,
                "index_creation_time": index_creation_time,
                "avg_exec": avg_exec,
                "exec_times": exec_times,
            })

        if not records:
            print(f"⚠️  No records to plot for source={source}, k={k}")
            return

        # Ensure output dir
        out_dir = os.path.join("./results", str(experiment), "plots")
        os.makedirs(out_dir, exist_ok=True)

        # Helpers ---------------------------------------------------------------
        def _fmt_s(v):     # format seconds nicely
            if v >= 1:
                return f"{v:.2f}s"
            return f"{v*1000:.0f}ms"

        def _annotate_bars(ax, rects, fontsize=9):
            for r in rects:
                h = r.get_height()
                ax.annotate(_fmt_s(h),
                            xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=fontsize)

        # NEW: stable method order + tiny x-jitter per method (to avoid perfect overlap)
        methods = [r["method"] for r in records]
        max_jitter = 0.12
        if len(methods) > 1:
            step = (2 * max_jitter) / (len(methods) - 1)
            jitter_vals = [-max_jitter + i * step for i in range(len(methods))]
        else:
            jitter_vals = [0.0]
        offsets = dict(zip(methods, jitter_vals))

        # --- 1) Line chart: per-query execution time, one line per method ---
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        max_len = 0
        for rec in records:
            y = rec["exec_times"]
            if not y:
                continue
            off = offsets.get(rec["method"], 0.0)
            x = np.arange(1, len(y) + 1, dtype=float) + off
            line, = plt.plot(x, y, marker="o", linewidth=1.5, label=rec["method"])
            max_len = max(max_len, len(y))

            # add dotted mean line in same color, jittered horizontally
            mean_val = float(np.mean(y))
            plt.hlines(mean_val, 1 + off, len(y) + off, linestyles=":", linewidth=1.5, colors=line.get_color())
            # annotate mean at end of its line segment
            ax1.annotate(f"mean={_fmt_s(mean_val)}", (len(y) + off, mean_val),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=line.get_color(), va="center")

        plt.title(f"Per-Query Execution Times - source={source}, k={k}")
        plt.xlabel("Query #")
        plt.ylabel("Execution time (s)")
        if max_len > 0:
            plt.xticks(np.arange(1, max_len + 1))  # keep integer query ticks
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        line_path = os.path.join(out_dir, f"{source}_k{k}_execution_{scenario}.png")
        plt.tight_layout()
        plt.savefig(line_path, dpi=150)
        plt.close()
        print(f"Saved line chart: {line_path}")

        # --- 2) Grouped bar chart: insertion, index per method ---
        insertion = [r["insertion_time"] for r in records]
        index_t   = [r["index_creation_time"] for r in records]

        x = np.arange(len(methods))
        width = 0.35

        plt.figure(figsize=(12, 6))
        ax2 = plt.gca()
        bars1 = ax2.bar(x - width/2, insertion, width, label="Insertion")
        bars2 = ax2.bar(x + width/2, index_t,   width, label="Index Creation")

        plt.title(f"Timing Summary by Method — source={source}, k={k}")
        plt.xlabel("Method")
        plt.ylabel("Time (s)")
        plt.xticks(x, methods, rotation=20, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend()

        # annotate bar heights
        _annotate_bars(ax2, bars1)
        _annotate_bars(ax2, bars2)

        bars_path = os.path.join(out_dir, f"{source}_k{k}_insertion_indexing_{scenario}.png")
        plt.tight_layout()
        plt.savefig(bars_path, dpi=150)
        plt.close()
        print(f"Saved bars chart: {bars_path}")

        # --- 3) Semantic coherence per query (line chart), + dotted mean per method ---
        from ResultReader import QueryResultsParser
        from sklearn.metrics.pairwise import cosine_similarity
        import ast

        plt.figure(figsize=(10, 6))
        ax3 = plt.gca()
        any_series = False

        for path, method in zip(test_files, methods):
            if not os.path.exists(path):
                continue
            try:
                parser = QueryResultsParser(path)
                qres = parser.query_results or {}

                # stable query order
                def _safe_int(x):
                    try: return int(x)
                    except: return x
                keys_sorted = sorted(qres.keys(), key=_safe_int)

                # per-query coherence (mean pairwise cosine over top-k tag texts)
                series = []
                for qid in keys_sorted:
                    items = qres[qid].get("results", [])
                    texts = []
                    for it in items:
                        t = it.get("tags")
                        texts.append(str(ast.literal_eval(t)) if isinstance(t, str) else str(t))
                    if len(texts) == 0:
                        series.append(0.0)
                    else:
                        emb = parser._st_model.encode(texts)       # (k,d)
                        sim_mat = cosine_similarity(emb, emb)      # kxk
                        mean = round(float(sim_mat.mean()),3)
                        series.append(mean)
                if not series:
                    continue

                off = offsets.get(method, 0.0)
                xq = np.arange(1, len(series) + 1, dtype=float) + off
                line, = plt.plot(xq, series, marker="o", linewidth=1.5, label=method)
                any_series = True

                # dotted mean line for this method (same color), jittered horizontally
                avg_coh = float(np.nanmean(series))
                plt.hlines(avg_coh, 1 + off, len(series) + off, linestyles=":", linewidth=1.5, colors=line.get_color())
                ax3.annotate(f"mean={avg_coh:.3f}", (len(series) + off, avg_coh),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=line.get_color(), va="center")

            except Exception:
                continue

        if any_series:
            plt.title(f"Per-Query Semantic Coherence — source={source}, k={k}")
            plt.xlabel("Query #")
            plt.ylabel(f"Semantic coherence@{k}")
            if 'keys_sorted' in locals() and len(keys_sorted) > 0:
                plt.xticks(np.arange(1, len(keys_sorted) + 1))
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            coh_path = os.path.join(out_dir, f"{source}_k{k}_coherence_{scenario}.png")
            plt.tight_layout()
            plt.savefig(coh_path, dpi=150)
            plt.close()
            print(f"Saved semantic coherence line chart: {coh_path}")
        else:
            print("⚠️  No coherence series to plot.")
        
        # --- 4) Spatial Range per query (line chart), + dotted mean per method ---
        plt.figure(figsize=(10, 6))
        ax4 = plt.gca()
        any_spatial = False
        _last_keys_sorted = None

        for path, method in zip(test_files, methods):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            qres = data.get("query_results", {})
            if not isinstance(qres, dict) or not qres:
                continue

            def _safe_int(x):
                try: return int(x)
                except: return x
            keys_sorted = sorted(qres.keys(), key=_safe_int)
            _last_keys_sorted = keys_sorted

            # compute spatial range = max distance among top-k
            series = []
            for qid in keys_sorted:
                items = qres[qid].get("results", []) or []
                dists = [float(it["distance"]) for it in items[:k] if "distance" in it]
                series.append(max(dists) if dists else np.nan)

            if all(np.isnan(series)):
                continue

            off = offsets.get(method, 0.0)
            xs = np.arange(1, len(series) + 1, dtype=float) + off
            arr = np.array(series, dtype=float)
            valid = ~np.isnan(arr)
            if not np.any(valid):
                continue

            line, = plt.plot(xs[valid], arr[valid], marker="o", linewidth=1.5, label=method)
            any_spatial = True

            mean_val = float(np.nanmean(arr))
            plt.hlines(mean_val, 1 + off, len(series) + off, linestyles=":", linewidth=1.5, colors=line.get_color())
            ax4.annotate(f"mean={mean_val:.2f}", (len(series) + off, mean_val),
                         xytext=(6, 0), textcoords="offset points",
                         fontsize=8, color=line.get_color(), va="center")

        if any_spatial:
            plt.title(f"Per-Query Spatial Range — source={source}, k={k}")
            plt.xlabel("Query #")
            plt.ylabel(f"Spatial range (metres)")
            if _last_keys_sorted:
                plt.xticks(np.arange(1, len(_last_keys_sorted) + 1))
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            rng_path = os.path.join(out_dir, f"{source}_k{k}_spatial_range_{scenario}.png")
            plt.tight_layout()
            plt.savefig(rng_path, dpi=150)
            plt.close()
            print(f"Saved spatial range line chart: {rng_path}")
        else:
            print("⚠️  No spatial range series to plot.")



    
def parse_args():
    p = argparse.ArgumentParser(
        description="Run result reader."
    )

    p.add_argument("--sce", choices=["all", "existing", "non_embedded", "embedded", "fused", "concat", "contrast"], default="embedded")
    p.add_argument("--exp", type=str, default=str(EXPERIMENT))
    p.add_argument("--so", nargs="+", default=SOURCE)
    p.add_argument("--k", nargs="+", type=int, default=K_VALUES)


    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    EXPERIMENT = args.exp
    SOURCE = args.so
    K_VALUES = args.k
    scenario_funcs = SCENARIOS[args.sce]

    print("=== Result Reader Config ===")
    print(f"Scenario     : {args.sce}")
    print(f"Experiment   : {args.exp}")
    print(f"Source(s)    : {args.so}")
    print(f"K Values     : {args.k}")
    print("============================")

 
    for source in SOURCE:
        for k in K_VALUES:
            print(f"source: {source}, k: {k} \n")
            test_files = []
            for method in scenario_funcs.keys():
                test_files.append(f"./results/{EXPERIMENT}/{source}_{method}_results_k{k}.json")
            for test_file in test_files:
                if not os.path.exists(test_file):
                    print(f"⚠️  Missing results file, skipping: {test_file}")
                    continue
                query_parser = QueryResultsParser(test_file)
                #query_parser.print_report(k=k)
            QueryResultsParser.produce_plots_for_group(test_files, source, k, EXPERIMENT, SCENARIO)