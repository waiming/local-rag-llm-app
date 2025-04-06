import os
import subprocess

if __name__ == "__main__":
    print("🚀 Launching Streamlit app...")
    os.environ["STREAMLIT_HOME"] = "src"
    subprocess.run(["streamlit", "run", "src/app_priming.py", "--server.runOnSave=false"])