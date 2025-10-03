# Base Class for datasets

import json
import os

class Dataset:
    def __init__(self, name, dataset_path):
        """
        Args:
            name: name of the dataset
            dataset_path: path to the dataset
        """
        self.name = name
        self.dataset_path = dataset_path
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        """
        Loads the dataset.
        Returns:
            data: the dataset
        """
        with open(os.path.join(self.dataset_path, "data.json"), 'r') as f:
            data = json.load(f)
        return data

    def format_prompt_basic(self, idx):
        """
        Formats a single prompt for the LLM.
        Args:
            idx: index of the example to use
        Returns:
            prompt: a formatted prompt for the LLM
        """
        raise NotImplementedError
    
    def format_prompt_concept_id(self, idx, concept_id_base_prompt_name, context_idx=0):
        with open(os.path.join(self.dataset_path, f"{concept_id_base_prompt_name}.txt"), "r") as f:  
            concept_id_few_shot_exemplar = f.read()
        instruction = concept_id_few_shot_exemplar
        instruction += self.format_question_info(idx, context_idx)
        instruction += "Concept List:\n"
        return instruction
    
    def format_prompt_concept_values(self, idx, concept_values_base_prompt_name, concepts, context_idx=0):
        with open(os.path.join(self.dataset_path, f"{concept_values_base_prompt_name}.txt"), "r") as f:  
            concept_values_few_shot_exemplar = f.read()
        instruction = concept_values_few_shot_exemplar
        instruction += self.format_question_info(idx, context_idx)
        instruction += "Concept List:\n"
        concept_str = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(concepts)])
        instruction += concept_str
        instruction += "\nConcept Values:\n"
        return instruction
    
    def format_prompt_counterfactual_gen(self, idx, counterfactual_base_prompt_name, concepts, intervene_bool, new_values, old_values, context_idx=0):
        with open(os.path.join(self.dataset_path, f"{counterfactual_base_prompt_name}.txt"), "r") as f:  
            counterfactual_few_shot_exemplar = f.read()
        instruction = counterfactual_few_shot_exemplar
        instruction += self.format_question_info(idx, context_idx)
        instruction += "Concept List:\n"
        concept_str = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(concepts)])
        instruction += concept_str
        instruction += "\nConcept Edits:\n"
        suffix = ""
        for idx in range(len(concepts)):
            concept = concepts[idx]
            concept = concept[:1].lower() + concept[1:] # lower case first letter
            if intervene_bool[idx]:
                if new_values[idx] == "UNKNOWN":
                    suffix+=f"{idx+1}. REMOVE: CHANGE from '{old_values[idx]}' to UNKNOWN\n"
                else:
                    suffix+=f"{idx+1}. CHANGE from '{old_values[idx]}' to '{new_values[idx]}'\n"
            else:
                suffix+=f"{idx+1}. KEEP\n"
        instruction += suffix
        return instruction

    def parse_counterfactual_output(self, counterfactual_output, includes_quality_checks=False):
        """
        Parses and validates the counterfactual output by checking that all expected entries are present and extracting them.
        Args:
            counterfactual_output: str, the counterfactual output to parse
            includes_quality_checks: bool, whether the outputs include quality check information
        Returns:
            output_dict: dict, a dictionary containing the parsed output
        """
        raise NotImplementedError
    
    def get_cot_answer_trigger(self, prompt=None, add_instr=None):
        """
        Returns the CoT answer trigger.
        Args:
            prompt: prompt to use for CoT answer trigger (ignored here because not needed in general case)
            add_instr: additional instructions to add to prompt
        """
        return f"{add_instr}\n\nLet's think step by step:"

    def get_direct_answer_trigger(self, prompt=None, add_instr=None):
        """
        Returns the direct answer trigger.
        Args:
            prompt: prompt to use for direct answer trigger (ignored here because not needed in general case)
            add_instr: additional instructions to add to prompt
        """
        return f"{add_instr}\n\n"

    def format_prompt_qa(self, basic_prompt, prompt_strategy, idx=None):
        """
        Formats a single prompt for the LLM for question answering.
        Args:
            basic_prompt: basic prompt to use for question answering
            prompt_strategy: prompting strategy to use
        Returns:
            prompt: a formatted prompt for the LLM
        """
        prompt = basic_prompt + f"""\n\n{self.get_cot_answer_trigger(basic_prompt, add_instr=prompt_strategy.add_instr) if prompt_strategy.cot 
                                         else self.get_direct_answer_trigger(basic_prompt, add_instr=prompt_strategy.add_instr)}"""
        if prompt_strategy.few_shot: # add few-shot examples to prompt
            with open(os.path.join(self.dataset_path, f"{prompt_strategy.few_shot_prompt_name}.txt"), "r") as f:
                few_shot_prompt = f.read()
            prompt = few_shot_prompt + prompt
        return prompt

    def format_question_counterfactual(self, counterfactual_dict):
        """
        Formats a single counterfactual question.
        Args:
            counterfactual_dict: dictionary containing the parts of the counterfactual question
        Returns:
            prompt: a formatted counterfactual question
        """
        raise NotImplementedError

    def format_prompt_qa_counterfactual(self, counterfactual_dict, prompt_strategy, idx=None):
        """
        Formats a single prompt for the LLM for question answering.
        Args:
            counterfactual_dict: dictionary containing the parts of the counterfactual question
            prompt_strategy: prompting strategy to use
            idx: index of the example to use
        Returns:
            prompt: a formatted prompt for the LLM
        """
        basic_prompt = self.format_question_counterfactual(counterfactual_dict)
        return self.format_prompt_qa(basic_prompt, prompt_strategy, idx=idx)


    def extract_answer(self, response, prompt_strategy, idx=None):
        """
      Extracts the answer from the LLM response.
        Args:
            response: LLM response
            prompt_strategy: prompting strategy used to extract response
            idx: index of the example that we're extracting the answer for
        Returns:
            pred: the predicted answer
        """
        raise NotImplementedError
    
    def get_answer_choices(self):
        """
        Returns the answer choices for a given question.
        Returns:
            answer_choices: the answer choices for the question
        """
        raise NotImplementedError
    