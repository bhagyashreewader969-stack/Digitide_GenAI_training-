import subprocess
import time

def run_ollama_prompt(model="llama3.2", prompt="Write a short poem about AI."):
    try:
        start = time.time()
        # Run ollama command
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, check=True
        )
        end = time.time()
        response_time = end - start

        print("=== Model Response ===")
        print(result.stdout.strip())
        print("\n=== Stats ===")
        print(f"Response Time: {response_time:.2f} seconds")

    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e.stderr)

if __name__ == "__main__":
    run_ollama_prompt()
