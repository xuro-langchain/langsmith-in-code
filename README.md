# LangSmith in Code

This repository contains comprehensive tutorials and examples for implementing evaluations and tracing in code for LangSmith. Enjoy!


## Pre-work

### Clone the LangSmith Workshops repo
```
git clone https://github.com/xuro-langchain/langsmith-in-code
```

### Create an environment and install dependencies  
```
# Ensure you have a recent version of pip and python installed
$ cd langsmith-in-code
$ uv sync
```

Create a .env file in the root repo folder using ```.env.example``` as an example.

## Running notebooks
Make sure the following command works and opens the relevant notebooks
```
$ jupyter notebook
```

The notebooks you should use are in the notebooks folder, which cover evaluations and tracing.
Evaluations are in the ```evaluations.ipynb``` notebook and Tracing is in the ```tracing.ipynb``` notebook.

## Adding CICD
There are two CICD modules included as part of this package
1. Running offline evaluations using Github Actions. The relevant files are
    * ```.github/workflows/evaluate.yml```: Contains the Github Action. 
        * Requires you to create a fork of this repo and to create an environment called ```production``` which you configure secrets on. 
        * This Github Action will trigger evaluations on pull request
    * ```cicd/test_evaluator.py```: Simple pytest evaluation that will run through the Github Action
    * ```cicd/report_eval.py```: Packages evaluation results to be attached to a pull request as a comment
2. Configuring Prompt Commit Webhooks. The relevant files are
    * ```cicd/prompthook.py```: Spins up a server to receive webhook notifications from LangSmith. 
        * Can be spun up locally by running ```uvicorn cicd.prompthook:app --reload``` in the root directory of this repo. 
        * Requires deployment to an external service like Render to connect to LangSmith [see detailed instructions here](https://docs.smith.langchain.com/prompt_engineering/tutorials/prompt_commit)
