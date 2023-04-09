#!/usr/bin/env python3
import os
import subprocess
import time
import torch
from collections import deque
from typing import Dict, List

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import sqlite3
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# Engine configuration

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
assert HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY environment variable is missing from .env"

HUGGINGFACE_API_MODEL = os.getenv("HUGGINGFACE_API_MODEL", "EleutherAI/gpt-neo-125m")
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

#PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
#assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

#PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
#assert (
#    PINECONE_ENVIRONMENT
#), "PINECONE_ENVIRONMENT environment variable is missing from .env"


# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "babyagi")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", "Develop a list of tasks"))

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

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions # but also provide command line
# arguments to override them

#if "gpt-4" in OPENAI_API_MODEL.lower():
#    print(
#        "\033[91m\033[1m"
#        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
#        + "\033[0m\033[0m"
#    )

# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")

# Connect to SQLite database
conn = sqlite3.connect("mydatabase.db")
c = conn.cursor()

# Create a table if it doesn't exist
table_name = YOUR_TABLE_NAME
c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)")

# Define a function to insert data into the table
def insert_data(text, embedding):
    c.execute(f"INSERT INTO {table_name} (text, embedding) VALUES (?, ?)", (text, embedding))
    conn.commit()

# Define a function to retrieve embeddings from the table
def get_embedding(text):
    if not text.strip():
        return None
    c.execute(f"SELECT embedding FROM {table_name} WHERE text=?", (text,))
    result = c.fetchone()
    if result is None:
        # Embed the text and insert it into the table
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy().tobytes()
        insert_data(text, embedding)
        return embedding
    else:
        return result[0]

# Task list
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if not text.strip():  # check if the input is empty or contains only whitespace
        return torch.zeros(1, 768)  # return a tensor of zeros
    input_ids = tokenizer(text, return_tensors='pt', truncation=True, padding=True)['input_ids']
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        embedding = model(input_ids)[0][:, 0, :]
    return embedding.numpy()


def openai_call(
    prompt: str,
    model: str = model_name,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            else:
                # Use chat completion pipeline
                if model.startswith("gpt-"):
                    model = f"gpt2-{model[4:]}"
                tokenizer = AutoTokenizer.from_pretrained(model)
                if "microsoft" in model:
                    model = AutoModelForCausalLM.from_pretrained(model, revision="main")
                    chat_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model)
                    chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
                response = chat_pipeline(prompt, max_length=max_tokens, temperature=temperature)
                return response[0]['generated_text'].strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split("\n")
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str) -> str:
    context = context_agent(query=objective, n=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    return openai_call(prompt, temperature=0.7, max_tokens=2000)


def context_agent(query: str, n: int):
    query_embedding = get_embedding(query)
    # Retrieve all embeddings from the database
    c.execute(f"SELECT text, embedding FROM {table_name}")
    rows = c.fetchall()
    # Calculate cosine similarity between query and all embeddings
    similarities = [(row[0], 1 - cosine(np.frombuffer(query_embedding), np.frombuffer(row[1]))) for row in rows]
    # Sort results by similarity
    sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Return top n tasks
    return [result[0] for result in sorted_results[:n]]

# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
	    namespace=OBJECTIVE
        )

        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again
