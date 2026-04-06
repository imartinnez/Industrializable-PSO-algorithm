if __name__ == "__main__":
    objectives = ["sphere", "rastrigin"]
    dims = [10]
    seeds = [1, 2, 3, 4, 5]
    mode = "sequential"

    common = {
        "max_iter": 2000,
        "strategy": "clamp",
        "fitness_policy": "plain",
        "topology": "global",
        "patience": 100,
        "imp_min": 1e-8,
        "tol": 1e-12,
    }

    param_grid = {
        "w": [0.4, 0.7, 0.9],
        "c1": [1.0, 1.5, 2.0],
        "c2": [1.0, 1.5, 2.0],
        "n_particles": [30, 50],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"grid_search_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for combo in grid_combinations(param_grid):
        print(f"Running combo: {combo}")
        rows = run_one_combo(
            combo=combo,
            objectives=objectives,
            dims=dims,
            seeds=seeds,
            mode=mode,
            common=common,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "grid_search_raw.csv", index=False)

    summary = (
        df.groupby(
            ["objective", "dim", "mode", "grid_w", "grid_c1", "grid_c2", "grid_n_particles"],
            as_index=False
        )
        .agg(
            mean_gap=("gap_to_optimum", "mean"),
            std_gap=("gap_to_optimum", "std"),
            mean_time=("total_time", "mean"),
            mean_iterations=("iterations_executed", "mean"),
        )
        .sort_values(["objective", "dim", "mean_gap", "mean_time"])
    )

    summary.to_csv(outdir / "grid_search_summary.csv", index=False)

    best = (
        summary.groupby(["objective", "dim", "mode"], as_index=False)
        .first()
    )
    best.to_csv(outdir / "grid_search_best.csv", index=False)

    config = {
        "objectives": objectives,
        "dims": dims,
        "seeds": seeds,
        "mode": mode,
        "common": common,
        "param_grid": param_grid,
    }

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nBest configs:")
    print(best)