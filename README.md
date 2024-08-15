# **Small LLM Tool-Use Pipeline**

This repository contains the code and resources for an approach to enhancing the tool-use capabilities of small language models (LLMs) such as Llama 7B. The method presented here improves upon existing pipelines by dynamically swapping in adapters trained on specific tool groupings, resulting in significant performance gains over previous methods. This approach is particularly effective in boosting the performance of small LLMs in natural language query answering tasks that require tool-use.

## **Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
3. [Commands](#commands)
4. [Examples](#examples)

## **Installation**

To set up this project, ensure you have a Linux environment with Conda installed. The environment should have high CUDA capabilities, as the vast majority of operations are computationally intensive. Additionally, the environment must have access to the internet during the setup process for downloading dependencies and generating the required datasets.

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Run the setup script to install all necessary dependencies, configure the environment, and generate the required datasets:
    ```bash
    ./setup.sh
    ```

## **Usage**

This project includes scripts to assist with training, inference, and evaluation for specific API tasks. The main scripts you'll interact with are:

- `train_api_creator.py`: Used to create job scripts for training models on specific APIs.
- `inference_patches_creator.py`: Used to generate job scripts for inference.
- `GLPFT/aggregateAdapters.py`: Used to merge adapters for inference.
- `GLPFT/exportCategories.py`: Exports the categories for the API families in the ToolBench dataset.
- `GLPFT/inference_utils/toolbench/evaluate-all-experts.py`: Evaluates model performance using specified inputs and outputs.
- `GLPFT/inference_utils/toolbench/gpt_inference.py`: Performs inference using GPT-3.5 Turbo through the OpenAI API, which can be easily edited in the code if desired.

## **Commands**

### **train_api_creator.py**

This script translates a Bash script into a Python script for training a model on specific APIs.

**Usage:**
```bash
python train_api_creator.py <api_name> [<model>] [<all_apis>]
```

**Example Commands:**

1. To create a training job script for the API `Data` using the `caller` model:
    ```bash
    python train_api_creator.py Data caller
    ```

2. To create a training job script for the API `chuck_norris` using the `backbone` model, with all APIs:
    ```bash
    python train_api_creator.py chuck_norris backbone True
    ```

3. To create a training job script for the API `free_nba` using the `caller[sports]` model:
    ```bash
    python train_api_creator.py free_nba caller[sports]
    ```

4. To create a training job script for the endpoint `exchange_for_currency_exchange` using the `llama` model:
    ```bash
    python train_api_creator.py exchange_for_currency_exchange llama
    ```

### **inference_patches_creator.py**

This script is used to generate job scripts for running inference on models with a specific suffix.

**Usage:**
```bash
python inference_patches_creator.py <model_suffix> [--trained_on_all]
```

**Example Commands:**

1. To generate a job script for the model `caller`:
    ```bash
    python inference_patches_creator.py caller
    ```

2. To generate a job script for the model `backbone` that was trained on all samples:
    ```bash
    python inference_patches_creator.py backbone --trained_on_all
    ```

3. To generate a job script for the model `caller[sports]`:
    ```bash
    python inference_patches_creator.py caller[sports]
    ```

### **GLPFT/aggregateAdapters.py**

This script is used to merge adapters for inference.

**Usage:**
```bash
python GLPFT/aggregateAdapters.py <model_suffix> [--trained_on_all]
```

**Example Commands:**

1. To merge adapters for the model `caller`:
    ```bash
    python GLPFT/aggregateAdapters.py caller
    ```

2. To merge adapters for the model `llama`, trained on all samples:
    ```bash
    python GLPFT/aggregateAdapters.py llama --trained_on_all
    ```

3. To merge adapters for the model `caller[sports]`:
    ```bash
    python GLPFT/aggregateAdapters.py caller[sports]
    ```

### **GLPFT/exportCategories.py**

This script exports the categories for the API families in the ToolBench dataset based on the documentation.

**Usage:**
```bash
python GLPFT/exportCategories.py
```

This script doesn't require any arguments.

### **GLPFT/inference_utils/toolbench/evaluate-all-experts.py**

This script evaluates model performance using specified input and output paths.

**Usage:**
```bash
python GLPFT/inference_utils/toolbench/evaluate-all-experts.py --input_path_folder <input_folder> --output_path <output_folder> --input_path_backoff <backoff_file> --input_path_gpt_backoff <gpt_backoff_file>
```

**Example Command:**

To evaluate the experts using the given paths:
```bash
python GLPFT/inference_utils/toolbench/evaluate-all-experts.py --input_path_folder output_patches/ --output_path output_patches/test --input_path_backoff output_verbose_res/predictions_caller.json --input_path_gpt_backoff output_verbose_res/gpt_full_outputs.json
```

### **GLPFT/inference_utils/toolbench/gpt_inference.py**

This script performs inference using GPT-3.5 Turbo through the OpenAI API in order to assess the performance of close-source models. The model can be easily edited in the code if desired.

**Usage:**
```bash
python GLPFT/inference_utils/toolbench/gpt_inference.py --start <start_index> --learning-mode <mode>
```

- `<start_index>`: The index of the entry to start processing from.
- `<mode>`: The learning mode, either `zero-shot` or `few-shot` (default: `zero-shot`).

**Example Command:**

To run inference in zero-shot mode starting from index 0:
```bash
python GLPFT/inference_utils/toolbench/gpt_inference.py --start 0 --learning-mode zero-shot
```

## **Examples**

### **Creating Training Job Scripts**

To create a training job script for various APIs:

- `Data` API using the `caller` model:
    ```bash
    python train_api_creator.py Data caller
    ```

- `chuck_norris` API using the `backbone` model, with all APIs:
    ```bash
    python train_api_creator.py chuck_norris backbone True
    ```

- `free_nba` API using the `caller[sports]` model:
    ```bash
    python train_api_creator.py free_nba caller[sports]
    ```

- `exchange_for_currency_exchange` endpoint using the `llama` model:
    ```bash
    python train_api_creator.py exchange_for_currency_exchange llama
    ```

### **Generating Inference Job Scripts**

To generate inference job scripts for different model configurations:

- For the `caller` model:
    ```bash
    python inference_patches_creator.py caller
    ```

- For the `backbone` model, trained on all samples:
    ```bash
    python inference_patches_creator.py backbone --trained_on_all
    ```

- For the `caller[sports]` model:
    ```bash
    python inference_patches_creator.py caller[sports]
    ```

### **Merging Adapters**

To merge adapters for different model configurations:

- For the `caller` model:
    ```bash
    python GLPFT/aggregateAdapters.py caller
    ```

- For the `llama` model, trained on all samples:
    ```bash
    python GLPFT/aggregateAdapters.py llama --trained_on_all
    ```

- For the `caller[sports]` model:
    ```bash
    python GLPFT/aggregateAdapters.py caller[sports]
    ```

### **Evaluating Experts**

To evaluate the experts using the specified input and output paths:
```bash
python GLPFT/inference_utils/toolbench/evaluate-all-experts.py --input_path_folder output_patches/ --output_path output_patches/test --input_path_backoff output_verbose_res/predictions_caller.json --input_path_gpt_backoff output_verbose_res/gpt_full_outputs.json
```

### **Running GPT Inference**

To run inference using GPT-3.5 Turbo in zero-shot mode starting from index 0:
```bash
python GLPFT/inference_utils/toolbench/gpt_inference.py --start 0 --learning-mode zero-shot
```
