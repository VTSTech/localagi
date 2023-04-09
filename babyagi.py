import os
import subprocess
import time
import torch
import transformers
from collections import deque
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import numpy as np
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Type

# Load default environment variables (.env)
load_dotenv()


# Engine configuration
# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
assert HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY environment variable is missing from .env"

HUGGINGFACE_API_MODEL = os.getenv("HUGGINGFACE_API_MODEL", "text-davinci-003")
assert HUGGINGFACE_API_MODEL, "HUGGINGFACE_API_MODEL environment variable is missing from .env"

model_name = HUGGINGFACE_API_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if "gpt-4" in HUGGINGFACE_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "babyagi")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", "Develop a task list"))
DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)

if ENABLE_COMMAND_LINE_ARGS:
    from extensions.argparseext import parse_arguments
    OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    from extensions.dotenvext import load_dotenv_extensions
    load_dotenv_extensions(DOTENV_EXTENSIONS)
# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")
print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
# Create an sklearn NearestNeighbors object with 1 neighbor
knn = NearestNeighbors(n_neighbors=1)
# Define a function to retrieve embeddings from the table
def get_embedding(prompt: str, knn: NearestNeighbors, model_name: str, dtype: Type[np.float32] = np.float32) -> bytes:
    model = transformers.AutoModel.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # encode the prompt using the model's tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # generate the embedding using the model
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    # remove the extra dimension from the embedding
    embedding = np.squeeze(embedding)

    # use the embedding to find the nearest neighbor
    distances, indices = knn.kneighbors([embedding])

    # return the embedding as bytes
    return embedding.tobytes()
# Create the NearestNeighbors index
print("\033[93m\033[1mCreating the NearestNeighbors index...\033[0m\033[0m")
dim = 768
index_params = {
    'n_jobs': -1,
    'algorithm': 'auto',
}
knn.set_params(**index_params)
space = knn.fit(np.zeros((1, dim)))
# Add the initial task embedding to the index
print("\033[93m\033[1mAdding the initial task embedding to the index...\033[0m\033[0m")
embedding = get_embedding(INITIAL_TASK, knn, model_name)
# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)
    print("\033[93m\033[1mAdding task...\033[0m\033[0m")
    #print(f"DEBUG: task: {task}")

# Define a function to retrieve a task from the task list
def get_task() -> Dict:
    if len(task_list) > 0:
        return task_list.popleft()
    else:
        # No tasks left in task list
        return None

# Define a function to create a task
def create_task(prompt: str) -> Dict:
    task = {
        "id": len(task_list) + 1,
        "prompt": prompt,
        "embedding": get_embedding(prompt, knn, model_name, dtype=np.float32),
        "status": "open"
    }
    print("\033[93m\033[1mCreating task...\033[0m\033[0m")
    print(f"DEBUG: prompt: {prompt}")
    return task

# Define a function to update a task in the task list
def update_task(task_id: int, status: str):
    for task in task_list:
        if task["id"] == task_id:
            task["status"] = status
            break

# Define a function to calculate the cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Define a function to find the most similar task to a given prompt
def find_most_similar_task(prompt: str) -> Dict:
    prompt_embedding = get_embedding(prompt, knn, model_name, dtype=np.float32)
    highest_similarity = -1
    most_similar_task = None
    for task in task_list:
        if task["status"] != "open":
            continue
        task_embedding = np.frombuffer(task["embedding"], dtype=np.float32)
        similarity = cosine_similarity(prompt_embedding, task_embedding)
        print(f"DEBUG: task: {task}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_task = task
    return most_similar_task
#Define a function to perform a task
def perform_task(task: Dict) -> Dict:
	print(f"\033[93m\033[1mPerforming task...\033[0m\033[0m")
	prompt = task["prompt"]
	response = pipeline(
		"text-generation",
		model=model_name,
		tokenizer=tokenizer,
		device_map="auto",
		batch_size=1
	)
	completion = response(prompt, max_length=30, temperature=0.5)[0]["generated_text"].strip()
	task["response"] = completion
	task["status"] = "completed"
	print(f"Task completed:\n{task}")
	return task

#Define the main function to run the program
def main():
    # Seed the task list with the initial task
    initial_task = create_task(INITIAL_TASK)
    add_task(initial_task)
    # Loop indefinitely until there are no more tasks
    print("DEBUG: Entering infinite loop")
    task_performed = False  # flag to check if any task has ever been performed
    while True:
        task = get_task()
        if task is None:
            # No more tasks
            break
        # Find the most similar task to the prompt
        most_similar_task = find_most_similar_task(task["prompt"])
        print(f"DEBUG: most_similar_task: {most_similar_task}")
        if most_similar_task is None and not task_performed:
            # No similar task found and no task has ever been performed, create a new one
            new_task = create_task(task["prompt"])
            add_task(new_task)
            perform_task(new_task)
            update_task(task["id"], "duplicate")
        elif most_similar_task is None and task_performed:
            # No similar task found but a task has already been performed, don't create a new one
            update_task(task["id"], "duplicate")
        else:
            # Perform the most similar task
            if not most_similar_task.get("performed", False):
                perform_task(most_similar_task)
                update_task(most_similar_task["id"], "completed")
            update_task(task["id"], "completed")
            task_performed = True  # set the flag to True


if __name__ == '__main__':
	main()
