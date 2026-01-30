import json
import re
import os

import pandas as pd

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
## Experiment Output Loading Utils ##
####################################################################################################


def load_intervention_information(example_idx, intervention_data_path):
    """
    Load intervention information from intervention data path for a single example.
    Args:
        example_idx: index of example
        intervention_data_path: path to intervention data
    Returns:
        concepts: list of concepts that interventions were applied to
        categories: list of categories associated with each concept
        concept_values: list of concept values
    """
    example_intervention_dir = os.path.join(intervention_data_path, f"example_{example_idx}")
    concept_file = os.path.join(example_intervention_dir, "concepts.json")
    assert os.path.exists(concept_file), f"Concept file not found: {concept_file}"
    with open(concept_file, "r") as f:
        concepts = json.load(f)
    categories_file = os.path.join(example_intervention_dir, "categories.json")
    assert os.path.exists(categories_file), f"Categories file not found: {categories_file}"
    with open(categories_file, "r") as f:
        categories = json.load(f)
    concept_values_file = os.path.join(example_intervention_dir, "concept_settings.json")
    assert os.path.exists(concept_values_file), f"Concept values file not found: {concept_values_file}"
    with open(concept_values_file, "r") as f:
        concept_values = json.load(f)
    return concepts, categories, concept_values


def load_original_model_responses(model_response_path, dataset_name, example_idx):
    """
    Load original model responses for example
    Args:
        model_response_path: path to model responses
        dataset_name: name of dataset
        example_idx: index of example
    Returns:
        dataframe with original model responses
    """
    response_dict = {"response_id": [], "prompt": [], "response": [], "answer": []}
    example_original_response_dir = os.path.join(model_response_path, f"example_{example_idx}", "original")
    for response_file in os.listdir(example_original_response_dir):
        assert response_file.startswith("response_n="), f"Invalid response file: {response_file}"
        with open(os.path.join(example_original_response_dir, response_file), "r") as f:
            response = json.load(f)
        response_id = int(response_file.split("response_n=")[1].split(".json")[0])
        response_dict["response_id"].append(f"original_n={response_id}")
        response_dict["prompt"].append(response["prompt"])
        response_dict["response"].append(response["response"])
        response_dict["answer"].append(response["answer"])
    return pd.DataFrame(response_dict)


def load_counterfactual_model_responses(model_response_path, example_idx, concepts, concept_values, categories):
    """
    Load counterfactual model responses for example
    Args:
        model_response_path: path to model responses
        example_idx: index of example
        concepts: list of concepts that interventions were applied to
        concept_values: list of concept values
        categories: list of categories associated with each concept
    Returns:
        dataframe with counterfactual model responses
    """
    response_dict = {"intrv_str": [], 
                        "intrv_bool": [],
                        "intrv_idx": [],
                        "intrv_concept": [],
                        "intrv_category": [],
                        "original_value": [],
                        "new_value": [],
                        "intrv_name": [],
                        "response_id": [], 
                        "prompt": [], 
                        "response": [], 
                        "answer": [], 
                        }
    example_counterfactual_response_dir = os.path.join(model_response_path, f"example_{example_idx}", "counterfactual")
    for response_file in os.listdir(example_counterfactual_response_dir):
        assert response_file.startswith("response_counterfactual="), f"Invalid response file: {response_file}"
        with open(os.path.join(example_counterfactual_response_dir, response_file), "r") as f:
            response = json.load(f)
        intervention_str = response_file.split("response_counterfactual=")[1].split("_n=")[0]
        completion_id = int(response_file.split("response_counterfactual=")[1].split("_n=")[1].split(".json")[0])
        response_dict["intrv_str"].append(intervention_str)
        response_dict["response_id"].append(f"counterfactual={intervention_str}_n={completion_id}")
        response_dict["prompt"].append(response["prompt"])
        response_dict["response"].append(response["response"])
        response_dict["answer"].append(response["answer"])
        intrv_bool, intrv_idx, intrv_concept, intrv_category, original_value, new_value, intrv_name = process_intervention_str(intervention_str, concepts, concept_values, categories)
        response_dict["intrv_bool"].append(intrv_bool)
        response_dict["intrv_idx"].append(intrv_idx)
        response_dict["intrv_concept"].append(intrv_concept)
        response_dict["intrv_category"].append(intrv_category)
        response_dict["original_value"].append(original_value)
        response_dict["new_value"].append(new_value)
        response_dict["intrv_name"].append(intrv_name)
    return pd.DataFrame(response_dict)


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


####################################################################################################
## Fine to Coarse Category Mapping ##
####################################################################################################

COARSE_CAT_MAP_BBQ = {
    "intention": "behavior",
    "quantity": "context",
    "race/ethnicity": "identity",
    "performance": "behavior",
    "location": "context",
    "gender": "identity",
    "time": "context",
    "action": "behavior",
    "sexual orientation": "identity",
    "personal belief": "behavior",
    "relationship": "context",
    "personality trait": "behavior",
    "personal commitment": "behavior",
    "topic": "context",
    "opinion": "behavior",
    "personal experience": "behavior",
    "experience": "behavior",
    "health": "identity",
    "social behavior": "behavior",
    "event": "context",
    "occupation": "identity",
    "physical condition": "identity",
    "physical ability": "identity",
    "object": "behavior",
    "appearance": "behavior",
    "gender identity": "identity",
    "treatment": "context",
    "situation": "context",
    "socioeconomic status": "identity",
    "possession": "behavior",
    "religion": "identity",
    "clothing": "behavior",
    "education": "context",
    "institution type": "context",
    "perspective": "behavior",
    "emotion": "behavior",
    "nationality": "identity",
    "name": "identity"
}

CONCEPT2CAT_CORRECT_BBQ = {
    'The purpose of the event': 'context',
    'The reason for choosing the restaurant': 'context',
    'The action taken by the police': 'context',
    'The reason the individuals were pulled over': 'context',
    'The relationship status of the individuals': 'behavior',
    'The health challenges mentioned': 'behavior',
    'The type of restaurant chosen': 'context'
}

# patient demographics categories
DEMOGRAPHIC_CATS_MEDQA = [
    "age", 
    "gender", 
    "race", 
    "sex",  
    "socioeconomic status"
]

# medical history & health background categories
HEALTH_BGRD_CATS_MEDQA = [
    "family history",
    "obstetric history",
    "allergy history",
    "medical history",
    'menstrual history',
    "pregnancy status",
    "health status",
    'vital signs history',
    'gestational age'
]

# clinical findings and diagnoses categories
CLINICAL_CATS_MEDQA = [
    "physical examination findings",
    "laboratory findings",
    "imaging findings",
    "biopsy findings",
    "pathology findings",
    "diagnostic tests",
    "neurological findings",
    "sensory findings",
    "vital signs",
    "neurologic findings",
    "respiratory symptoms",
    "urinary/bowel symptoms",
    "diganois",
    "initial diagnosis",
    "diagnostic test findings",
    "diagnostic test results",
    "negative findings",
    "pregnancy outcome",
    'diagnosis',
    'audiometry findings',
    'neurological assessment',
    'ophthalmology findings'
]

# symptoms categories
SYMPTOM_CATS_MEDQA = [
    "symptom location",
    "symptom impact",
    "symptom duration",
    "symptoms",
    "symptom triggers",
    "pain characteristics",
    "symptom timeline",
    "symptom characteristics",
    "symptom response",
    "review of systems"
]

# treatment and mangagement categories
TREATMENT_CATS_MEDQA = [
    "medication",
    "treatment recommendation",
    "treatment",
    "pre-hospital treatment",
    "treatment refusal",
    "treatment plan",
    "medication compliance",
    "medication adherence",
    "prenatal care",
    "clinical course",
    'treatment response'
]

# behavior and psychological factors categories
BEHAVIORAL_CATS_MEDQA = [
    "mental status",
    "behavioral change",
    "behavioral response",
    "mental health",
    "mental health history",
    "cognitive function",
    "social behavior",
    "self-perception",
    "sleep quality",
    "physical appearance",
    "mobility",
    "weight change",
    "third-party observations",
    "patient concerns",
    "reason for visit",
    "social history", 
    "lifestyle",
    "travel history", 
    "family dynamics",
    "environmental factors",
    "physical characteristics",
    'sexual history',
    'mental health status',
    'BMI',
    'occupation',
    'substance use'
]

COARSE_CATEGORY_MAPPING_INV_MEDQA = {
    "demographics": DEMOGRAPHIC_CATS_MEDQA,
    "health background": HEALTH_BGRD_CATS_MEDQA,
    "clinical": CLINICAL_CATS_MEDQA,
    "symptoms": SYMPTOM_CATS_MEDQA,
    "treatment": TREATMENT_CATS_MEDQA,
    "behavioral": BEHAVIORAL_CATS_MEDQA
}

COARSE_CATEGORY_MAPPING_MEDQA = {v: k for k, values in COARSE_CATEGORY_MAPPING_INV_MEDQA.items() for v in values}

def apply_coarse_cat_mapping_to_df(df, dataset_name,coarse_cat_name="intrv_category_coarse"):
    if dataset_name == "bbq":
        df[coarse_cat_name] = df["intrv_category"].apply(lambda x: COARSE_CAT_MAP_BBQ.get(x, x))
        for concept, cat in CONCEPT2CAT_CORRECT_BBQ.items():
            df.loc[(df["intrv_concept"] == concept), coarse_cat_name] = cat
    elif dataset_name == "medqa":
        df[coarse_cat_name] = df["intrv_category"].apply(lambda x: COARSE_CATEGORY_MAPPING_MEDQA.get(x, x))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return df


####################################################################################################
## Miscellaneous Utils ##
####################################################################################################

def process_intervention_str(intrv_str, concept, concept_values, categories):
    intrv_bool = [x != "0" for x in intrv_str]
    intrv_idx = intrv_bool.index(True)
    intrv_concept = concept[intrv_idx]
    intrv_category = categories[intrv_idx]
    original_value = concept_values[intrv_idx]["current_setting"]
    intrv_char = intrv_str[intrv_idx]
    assert len(concept_values[intrv_idx]["new_settings"]), "Current method handles only a single alternative value for each concept."
    new_value = "UNKNOWN" if intrv_char == '-' else concept_values[intrv_idx]["new_settings"][0]
    intrv_name = f"{intrv_concept}: {original_value} -> {new_value}"
    return intrv_bool, intrv_idx, intrv_concept, intrv_category, original_value, new_value, intrv_name
    