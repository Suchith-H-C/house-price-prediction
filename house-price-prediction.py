import subprocess

if __name__ == "__main__":
    subprocess.run(["python", "src/data_preprocessing.py"], check=True)
    subprocess.run(["python", "src/training.py"], check=True)
    subprocess.run(["python", "src/evaluate.py"], check=True)
    print("✅ All steps completed: Preprocessing, Training, and Evaluation.")
