# Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations
Official implementation of [Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations](https://openreview.net/forum?id=4ub9gpx9xw) accepted at ICLR 2025.

Katie Matton, Robert Ness, John Guttag, Emre Kıcıman

## Overview

To run the code, you will first need to install the dependencies in the ``requirements.txt`` file.

Our code uses the OpenAI and Anthropic APIs to query LLMs. To do this, you will need to create an account and provide your API key. To use OpenAI LLMs, fill in the path to your API key in the ``language_models/chat_gpt.py`` and ``language_models/completion_gpt.py`` files. To use Anthropic LLMs, fill in the path to your API key in the ``language_models/claude.py`` file.

There are two key parts to our method: (1) collecting counterfactual data and (2) estimating faithfulness using the collected data. We have examples for how to run each step of our method in the ``notebooks`` directory.

For collecting counterfactual data, there are three parts:
* To extract concepts and generate counterfactuals, use ``src/run_generate_interventions.py``.
* To collect model responses to both the original and counterfactual questions, use ``src/run_collect_model_responses.py``.
* To analyze the model's explanations (to determine which concepts it implies influenced its answer choice), use ``src/run_determine_implied_concepts.py``.

To estimate faithfulness, we carried out all steps in Jupyter notebooks. To replicate our faithfulness estimation results on the BBQ dataset, see ``notebooks/bbq_example.ipynb``. To replicate our faithfulness estimation results on the MedQA dataset, see ``notebooks/medqa_example.ipynb``. The relevant helper code is in the directories ``src/explanation_implied_effect_estimation``, ``src/causal_concept_effect_estimation``, and ``src/faithfulness_estimation``.


### Data

We use the variant of the BBQ dataset ([Parrish et al., ACL 2022](https://aclanthology.org/2022.findings-acl.165/)) introduced in ([Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388)). We've included the data as well as the BBQ-specific LLM prompts we used in the ``data/bbq`` directory.

We use the MedQA dataset [(Jin et al., Applied Sciences 2021)](https://www.mdpi.com/2076-3417/11/14/6421). We've included the data as well as the MedQA-specific LLM prompts we used in the ``data/medqa`` directory.

### Examples

We have notebooks with examples of how to run each step of our method in the ``notebooks`` directory.
