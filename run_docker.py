import subprocess
import os
from dotenv import load_dotenv

IMAGE_NAME = "mlops-project2"

def main():
    print("Loading WANDB_API_KEY...")
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        print("Error: WANDB_API_KEY not found.")
        return

    try:
        print(f"Building Docker image: {IMAGE_NAME}")
        subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)
        print("Docker build complete")

        print(f"Running Docker container: {IMAGE_NAME}")
        run_command = ["docker", "run", "--rm", "-e", f"WANDB_API_KEY={api_key}", IMAGE_NAME]
        subprocess.run(run_command, check=True)
        print("Container run complete")

    except subprocess.CalledProcessError as e:
        print(f"Error during Docker process (exit code {e.returncode})")
    except FileNotFoundError:
        print("Error: 'docker' command not found")

if __name__ == "__main__":
    main()
