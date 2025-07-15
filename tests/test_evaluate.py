import sys
import os
import pytest
import yaml

# Add 'src' directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from evaluate import evaluate  # now this will work

def test_evaluate_runs_successfully(tmp_path):
    """
    Test that evaluate() runs without raising exceptions
    and generates the expected output files.
    """
    with open("params.yml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_file = "evaluation_metrics.txt"
    if os.path.exists(eval_file):
        os.rename(eval_file, eval_file + ".bak")

    try:
        evaluate(cfg)
        assert os.path.exists(eval_file), "evaluation_metrics.txt was not created"
        with open(eval_file, "r") as f:
            content = f.read()
            assert "RMSE" in content
            assert "MAE" in content
            assert "R2" in content
    finally:
        if os.path.exists(eval_file):
            os.remove(eval_file)
        if os.path.exists(eval_file + ".bak"):
            os.rename(eval_file + ".bak", eval_file)
