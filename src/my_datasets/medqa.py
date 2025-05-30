import json
import os

from my_datasets.dataset import Dataset
from my_datasets.utils import get_options_in_str_medqa, extract_context_and_final_question

class MedQADataset(Dataset):
    def __init__(self, name, dataset_path):
        super().__init__(name, dataset_path)

    def load_data(self):
        """
        Loads the dataset.
        Note: we have a separate function because MedQA needs to be loaded with utf-8 encoding.
        Returns:
            data: the dataset
        """
        with open(os.path.join(self.dataset_path, "data.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_prompt_basic(self, idx, context_idx=None, double_space=False):
        """
        Formats a single MedQA question for the LLM in most basic format (without CoT, few shot examples, etc.).
        Args:
            idx: index of the question
        Returns:
            prompt: a formatted prompt for the LLM
        """
        row = self.data[idx]
        question = row['question']
        options = get_options_in_str_medqa(row)
        prompt = f"""Question: {question}\n{options}"""
        return prompt
    
    def format_question_info(self, idx, context_idx=None):
        row = self.data[idx]
        context, question = extract_context_and_final_question(row['question'])
        options = get_options_in_str_medqa(row)
        question_info = f"Context: {context}\n"
        question_info += f"Question: {question}\n"
        question_info += f"Answer choices:\n{options}\n"
        return question_info
    
    def parse_counterfactual_output(self, counterfactual_output):
        """
        Parses and validates the counterfactual output by checking that all expected entries are present and extracting them.
        Args:
            counterfactual_output: str, the counterfactual output to parse
        Returns:
            output_dict: dict, a dictionary containing the parsed output
        """
        output_dict = {}
        counterfactual_output_lines = counterfactual_output.split("\n")
        line_idx = 0
        if not counterfactual_output_lines[line_idx]:
            line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("Edited Context:"), f"Edited Context not found in counterfactual output where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_context"] = counterfactual_output_lines[line_idx].split("Edited Context:")[1].strip()
        line_idx += 1
        if not counterfactual_output_lines[line_idx]:
            line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("Edited Question:"), f"Edited Question not found in counterfactual output where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_question"] = counterfactual_output_lines[line_idx].split("Edited Question:")[1].strip()
        line_idx += 1
        if not counterfactual_output_lines[line_idx]:
            line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("Edited Answer choices:"), f"Edited Answer Choices not found in counterfactual output where expected. Counterfactual output: {counterfactual_output}"
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("A."), f"Edited Answer Choice A. not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans0"] = counterfactual_output_lines[line_idx].split("A.")[1].strip()
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("B."), f"Edited Answer Choice B. not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans1"] = counterfactual_output_lines[line_idx].split("B.")[1].strip()
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("C."), f"Edited Answer Choice C. not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans2"] = counterfactual_output_lines[line_idx].split("C.")[1].strip()
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("D."), f"Edited Answer Choice D. not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans3"] = counterfactual_output_lines[line_idx].split("D.")[1].strip()
        line_idx += 1
        if not counterfactual_output_lines[line_idx]:
            line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("Comments on coherency:"), f"Comments on coherency not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["coherency_comments"] = counterfactual_output_lines[line_idx].split("Comments on coherency:")[1].strip()
        line_idx += 1
        if not counterfactual_output_lines[line_idx]:
            line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("Coherent YES/NO:"), f"Coherency decision not where expected. Counterfactual output: {counterfactual_output}"
        assert counterfactual_output_lines[line_idx].split(":")[1].strip() in ["YES", "NO"], f"Coherency decision not YES or NO. Counterfactual output: {counterfactual_output}"
        output_dict["coherent"] = counterfactual_output_lines[line_idx].split(":")[1].strip()
        return output_dict
