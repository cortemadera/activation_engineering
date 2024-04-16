# Shakespearean Summaries

This repo uses the mistralai/Mistral-7B-Instruct-v0.2 model to produce summaries of news articles in Shakespearean style, including experimenting with the activation engineering approach (https://arxiv.org/pdf/2308.10248.pdf).

## Stack
* Mistral-7B-Instruct-v0.2
* transformer_lens-1.15.0

## File Descriptions
1. standard_summary.py: Generate summary in modern English

2. shakespearean_summary_baseline.py: Generate summary in Shakespearean style using one-shot prompting, not utilizing activation engineering

3. shakespearean_summary_activation.py: Generate summary in Shakespearean style using activation engineering

## How to Run
1. Install transformer_lens from local (current release only supports mistralai/Mistral-7B-Instruct-v0.1, not v0.2)

* cd transformer_lens-1.15.0
* poetry install
* pip install -e .

2. Install torch and transformers

* cd ..
* pip install -r requirements.txt

## Results
Reference results are in results.txt

## Thoughts on Eval
1. If human-crafted reference summaries are available, ROUGE score can be a resonably reliable evaluation, but reference summaries are also very hard to scale.

2. It may be feasible to use an LLM for evaluation. For example, one could use a prompt similar to that in eval_prompt.jinja.

3. I experimented with DeepEval’s summarization evaluation. The [idea](https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task) is quite appealing, however, my low-volume experiments show results are quite unreliable at this stage.

## Thoughts on Activation Engineering Experiments

Compared to the one-shot prompt baseline approach for creating Shakespearean summaries, I had great difficulty steering the model with activation engineering. 

The same steering prompt pair is instantly effective for simple base prompts like 'I went up to my friend and said', but quite ineffective for the summarization prompt. Details are commented in shakespearean_summary_activation.py for later review.