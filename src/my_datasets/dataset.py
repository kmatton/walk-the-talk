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
    