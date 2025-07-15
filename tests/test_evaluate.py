import pytest
import yaml
from evaluate import evaluate
import os
import shutil

def test_evaluate_runs_successfully():
    """
    Test that evaluate() runs successfully and generates expected output file.
    """

    # Load config
    with open("params.yml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_file = "evaluation_metrics.txt"
    backup_file = eval_file + ".bak"

    # Backup existing output file (if any)
    if os.path.exists(eval_file):
        shutil.copy(eval_file, backup_file)

    try:
        evaluate(cfg)
        assert os.path.exists(eval_file), "evaluation_metrics.txt was not created"

        with open(eval_file, "r") as f:
            content = f.read()
            assert "RMSE" in content
            assert "MAE" in content
            assert "R2" in content
            assert "MSE" in content

    finally:
        # Clean up
        if os.path.exists(eval_file):
            os.remove(eval_file)
        if os.path.exists(backup_file):
            shutil.move(backup_file, eval_file)
