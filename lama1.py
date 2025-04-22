import subprocess

def run_ollama(prompt):
    # Command to run Ollama with the desired model
    command = ["ollama", "run", "mistral-nemo"]
    
    # Start the process and pass the prompt to Ollama
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    output, error = process.communicate(input=prompt)

    # Check for errors
    if process.returncode != 0:
        print("Error running Ollama:", error)
        return None
    
    return output

# Remove or comment out the while loop
# if __name__ == "__main__":
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = run_ollama(user_input)
#         print("Ollama:", response)
