RUN_SPEC = {
    "name": "nycc_full",
    "verbose": True,
    "models": {
        "base": "hf_local:/workspace/models/qwen3.6-27b",
        "prompted": "hf_local:/workspace/models/qwen3.6-27b",
        "NYCC-trained": "hf_local:/workspace/models/qwen3.6-27b-nycc-vllm",
        "gpt-4.1": "openai/gpt-4.1",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    },
    "dataset": {"path": "data/scenarios/nycc_scenarios.json", "start": 0, "count": 30, "shuffle": False},
    "constitution": {"path": "data/constitutions/nycc.json", "num_criteria": 10},
    "collection": {
        "enabled": False,
        "evaluations_path": "/workspace/EigenBench/data/responses/nycc_evaluations.jsonl",
        "allow_ties": True,
    },
    "training": {
        "enabled": True, "model": "btd_ties", "dims": [2], "lr": 1e-3,
        "weight_decay": 0.0, "max_epochs": 1000, "batch_size": 32, "device": "cpu",
        "test_size": 0.2, "group_split": False, "separate_criteria": False,
        "bootstrap": {"enabled": True, "n_bootstraps": 100, "random_seed": 42,
                      "save_models": False, "save_trust_matrices": True},
    },
}
