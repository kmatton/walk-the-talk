import re

from my_datasets.bbq import BBQDataset
from my_datasets.medqa import MedQADataset
from language_models.chat_gpt import ChatGPT
from language_models.claude import Claude
from language_models.completion_gpt import CompletionGPT


####################################################################################################
## LLM Response Parsing Utils ##
####################################################################################################


def parse_llm_response_concepts_and_categories(response):
    """
    Parses the response from the LLM for concept identification.
    Args:
        response: response from the LLM
    Returns:
        concepts: a list of concepts identified by the LLM
        categories: a list of the category associated with each concept
    """
    response_lines = response.strip().split("\n")
    concepts = []
    categories = []
    # check that factors match desired format
    for idx, line in enumerate(response_lines):
        if not line.startswith(str(idx+1)):
            raise ValueError(f"Concept ({idx+1}) {line} does not match expected format. Full response was {response}")
        try:
            concept, category = line.split(" (Category = ")
        except ValueError:
            raise ValueError(f"Concept ({idx+1}) {line} does not match expected format for category extraction.")
        concepts.append(concept.strip()[3:].strip()) # remove leading number
        categories.append(category[:-1].strip()) # remove trailing parenthesis
    return concepts, categories


def parse_llm_response_factor_settings(response):
    """
    Parses the response from the LLM for identifying current/alternative settings of each factor.
    Args:
        response: response from the LLM
    Returns:
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
    """
    response_lines = response.strip().split("\n")
    factor_settings = []
    for idx, line in enumerate(response_lines):
        if not line.startswith(str(idx+1)):
            raise ValueError(f"Concept Values {idx+1} ({line}) does not match expected format. Full response was {response}")
        line = line.strip()[3:].strip()
        if not line.startswith('(A)') or ('(B.1)' not in line and '(B)' not in line):
            raise ValueError(f"Concept Values at line {idx+1} ({line}) do not match expected format.")
        if '(B.3)' in line:
            raise ValueError(f"Concept Values at line {idx+1} ({line}) do not match expected format (parsing does not handle more than 2 alternative values currently).")
        if '(B.1)' in line:
            current_setting = line.split('(B.1)')[0].split('(A)')[1].strip()
        else:
            current_setting = line.split('(B)')[0].split('(A)')[1].strip()
        if '(B.2)' in line:
            new_settings = [line.split('(B.1)')[1].split('(B.2)')[0].strip(), line.split('(B.2)')[1].strip()]
        elif '(B.1)' in line:
            new_settings = [line.split('(B.1)')[1].strip()]
        else:
            new_settings = [line.split('(B)')[1].strip()]
        factor_settings.append({"current_setting": current_setting, "new_settings": new_settings})
    return factor_settings


def parse_llm_response_implied_concepts(response, n_concepts):
    """
    Parses the LLM response about which concepts are implied by the LLM's explanation.
    Args:
        response: response from the LLM
        n_concepts: the number of concepts we expect the LLM to provide decisions for
    Returns:
        parsed_fds: a list of 1s and 0s indicating whether each concept is implied by the CoT explanation
    """
    # split based on presence of numbers followed by a period and a space
    pattern = re.compile(r'\d+\.\s')
    concept_decisions = pattern.split(response)[1:]
    if len(concept_decisions) != n_concepts:
        raise ValueError(f"Number of concept decisions does not match expected number of concepts. Expected {n_concepts}, got {len(concept_decisions)}. Full response was {response}.")
    parsed_fds = []
    for idx, concept_decision in enumerate(concept_decisions):
        decision_bools = ["YES" in concept_decision, "NO" in concept_decision]
        if sum(decision_bools) != 1:
            raise ValueError(f"Concept decision {idx+1} does not match expected format. (Did not provide yes/no decision). Full response was {response}.")
        parsed_fds.append(1 if "YES" in concept_decision else 0)
    return parsed_fds, response


####################################################################################################
## Intervention Generation Utils ##
####################################################################################################

def enumerate_interventions_helper(intervention_list, intervention_str, factors, factor_settings, k_hop):
    """
    Helper function for enumerating all possible interventions.
    Args:
        intervention_list: a list of intervention vectors
        intervention_str: a string representing a choice of interventions
        factors: a list of factors to intervene on
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
        k_hop: if not None, only enumerate interventions that are k hops away from the original prompt
    """
    if len(intervention_str) == len(factors):
        intervention_list.append(intervention_str)
    elif sum([intervention_str[i] != "0" for i in range(len(intervention_str))]) == k_hop:
        # add 0s as remaining digits
        intervention_list.append(intervention_str + "0" * (len(factors) - len(intervention_str)))
    else:
        # no intervention case
        enumerate_interventions_helper(intervention_list, intervention_str + "0", factors, factor_settings, k_hop)
        # loop over possible interventions
        for idx in range(len(factor_settings[len(intervention_str)]["new_settings"])):
            enumerate_interventions_helper(intervention_list, intervention_str + str(idx+1), factors, factor_settings, k_hop)


def enumerate_interventions(factors, factor_settings, k_hop=None, include_no_intervention=True, mark_removals=True):
    """
    Enumerates all possible interventions.
    Args:
        factors: a list of factors to intervene on 
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
        k_hop: if not None, only enumerate interventions that are k hops away from the original prompt
        include_no_intervention: whether to include the no intervention case
        mark_removals: whether to mark removal interventions with a special symobol
    """
    intervention_list = []
    enumerate_interventions_helper(intervention_list, "", factors, factor_settings, k_hop)
    no_intrv_str = "0" * len(factors)
    if not include_no_intervention and no_intrv_str in intervention_list:
        intervention_list.remove(no_intrv_str)
    if mark_removals:
        for idx in range(len(intervention_list)):
            intrv_str = intervention_list[idx]
            for j in range(len(intrv_str)):
                if intrv_str[j] != "0" and factor_settings[j]['new_settings'][int(intrv_str[j]) - 1] == "UNKNOWN":
                    intrv_str = intrv_str[:j] + "-" + intrv_str[j+1:]
            intervention_list[idx] = intrv_str
    return intervention_list


####################################################################################################
## Class Factory Helper Functions ##
####################################################################################################

def get_language_model(model_name, max_tokens=256, temperature=0.7):
    if 'gpt-4' in model_name or model_name == "gpt-3.5-turbo-0613":
        return ChatGPT(model_name, temperature=temperature)
    elif model_name == 'text-davinci-003' or model_name == 'gpt-3.5-turbo-instruct':
        return CompletionGPT(model_name, max_tokens=max_tokens, temperature=temperature)
    elif 'claude'in model_name:
        return Claude(model_name, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

def get_dataset(dataset_name, dataset_path):
    if dataset_name == "bbq":
        return BBQDataset(dataset_name, dataset_path)
    elif dataset_name == "medqa":
        return MedQADataset(dataset_name, dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
