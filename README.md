<h1 align="center">
 babyagi

</h1>

# Objective
This Python script is an example of an AI-powered task management system. The system uses HuggingFace Models and an SQLite database to create, prioritize, and execute tasks. The main idea behind this system is that it creates tasks based on the result of previous tasks and a predefined objective.

This README will cover the following:

* [How the script works](#how-it-works)

* [How to use the script](#how-to-use)

* [Supported Models](#supported-models)

* [Warning about running the script continuously](#continous-script-warning)
# How It Works<a name="how-it-works"></a>
The script works by running an infinite loop that does the following steps:

1. Pulls the first task from the task list.
2. Sends the task to the execution agent, which uses HuggingFace Language Model to complete the task based on the context.
3. Enriches the result and stores it in SQLite3.
4. Creates new tasks and reprioritizes the task list based on the objective and the result of the previous task.
The execution_agent() function is where the H API is used. It takes two parameters: the objective and the task. It then sends a prompt to OpenAI's API, which returns the result of the task. The prompt consists of a description of the AI system's task, the objective, and the task itself. The result is then returned as a string.

The task_creation_agent() function is where HuggingFace Model is used to create new tasks based on the objective and the result of the previous task. The function takes four parameters: the objective, the result of the previous task, the task description, and the current task list. It then sends a prompt to HuggingFace Model, which returns a list of new tasks as strings. The function then returns the new tasks as a list of dictionaries, where each dictionary contains the name of the task.

The prioritization_agent() function is where HuggingFace Model is used to reprioritize the task list. The function takes one parameter, the ID of the current task. It sends a prompt to HuggingFace Model, which returns the reprioritized task list as a numbered list.

Finally, the script uses Pinecone to store and retrieve task results for context. The script creates a Pinecone index based on the table name specified in the YOUR_TABLE_NAME variable. Pinecone is then used to store the results of the task in the index, along with the task name and any additional metadata.

# How to Use<a name="how-to-use"></a>
To use the script, you will need to follow these steps:

1. Clone the repository via `git clone https://github.com/Veritas83/localagi.git` and `cd` into the cloned repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Copy the .env.example file to .env: `cp .env.example .env`. This is where you will set the following variables.
4. Set your HuggingFace API key in the HUGGINGFACE_API_KEY variables.
6. Set the name of the table where the task results will be stored in the TABLE_NAME variable.
7. (Optional) Set the objective of the task management system in the OBJECTIVE variable.
8. (Optional) Set the first task of the system in the INITIAL_TASK variable.
9. Run the script.

All optional values above can also be specified on the command line.

# Supported Models<a name="supported-models"></a>

This script works with all HuggingFace models, as well as Llama through Llama.cpp. Default model is **gpt-3.5-turbo**. To use a different model, specify it through HUGGINGFACE_API_MODEL or use the command line.

## Llama

Download the latest version of [Llama.cpp](https://github.com/ggerganov/llama.cpp) and follow instructions to make it. You will also need the Llama model weights.

 - **Under no circumstances share IPFS, magnet links, or any other links to model downloads anywhere in this repository, including in issues, discussions or pull requests. They will be immediately deleted.**

After that link `llama/main` to llama.cpp/main and `models` to the folder where you have the Llama model weights. Then run the script with `OPENAI_API_MODEL=llama` or `-l` argument.

# Warning<a name="continous-script-warning"></a>
This script is designed to be run continuously as part of a task management system. Running this script continuously can result in high API usage, so please use it responsibly.

# Contribution
Needless to say, BabyAGI is still in its infancy and thus we are still determining its direction and the steps to get there. Currently, a key design goal for BabyAGI is to be *simple* such that it's easy to understand and build upon. To maintain this simplicity, we kindly request that you adhere to the following guidelines when submitting PRs:
* Focus on small, modular modifications rather than extensive refactoring.
* When introducing new features, provide a detailed description of the specific use case you are addressing.

# Backstory
BabyAGI is a pared-down version of the original [Task-Driven Autonomous Agent](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20) (Mar 28, 2023) shared on Twitter. This version is down to 140 lines: 13 comments, 22 blanks, and 105 code. The name of the repo came up in the reaction to the original autonomous agent - the author does not mean to imply that this is AGI.
