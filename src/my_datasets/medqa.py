import json
import os
import re

from my_datasets.dataset import Dataset
from my_datasets.utils import get_options_in_str_medqa, extract_context_and_final_question, few_shot_rank_cot_knn_md

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

    def get_cot_answer_trigger(self, prompt=None, add_instr=None):
        """
        Returns the CoT answer trigger.
        Args:
            prompt: prompt to use for CoT answer trigger (ignored here because not needed in general case)
            add_instr: additional instructions to add to prompt
        """
        if add_instr is not None:
            prompt = f"{add_instr}\n\n"
        else:
            prompt = ""
        return prompt + "Explanation: Let's solve this step-by-step, referring to authoritative sources as needed."

    def format_prompt_qa(self, basic_prompt, prompt_strategy, idx=None):
        """
        Formats a single prompt for the LLM for question answering.
        Args:
            basic_prompt: basic prompt to use for question answering
            prompt_strategy: prompting strategy to use
            idx: idx of example to use
        Returns:
            prompt: a formatted prompt for the LLM
        """
        if prompt_strategy.knn_rank:
            prompt = few_shot_rank_cot_knn_md(self.data[idx])
        else:
            prompt = super().format_prompt_qa(basic_prompt, prompt_strategy)
            prompt_prefix = "You are a medical expert. Your task is to answer multiple choice questions about medical knowledge.\n\n###\n\n"
            prompt = prompt_prefix + prompt
            prompt += " Please make sure that the last line of your answer is in the form 'Answer: [A/B/C/D]'."
        return prompt

    def format_question_counterfactual(self, counterfactual_dict, double_space=False):
        prompt = f"""Question: {counterfactual_dict["edited_context"]} {counterfactual_dict["edited_question"]}\nA. {counterfactual_dict["edited_ans0"]}\nB. {counterfactual_dict["edited_ans1"]}\nC. {counterfactual_dict["edited_ans2"]}\nD. {counterfactual_dict["edited_ans3"]}"""
        return prompt

    def extract_answer(self, response, prompt_strategy, idx=None):
        choices = "ABCD"
        pred = None
        response_last_line = response.strip().split("\n")[-1]
        if prompt_strategy.knn_rank or prompt_strategy.cot:
            # check if only one of the answer choices is in the response
            found1 = [re.search(f"\({choice}\)", response) is not None for choice in choices]
            found2 = [re.search(f"{choice}\.", response) is not None for choice in choices]
            if sum(found1) == 1:
                # this is the answer
                pred = choices[found1.index(True)]
            elif sum(found2) == 1:
                # this is the answer
                pred = choices[found2.index(True)]
            elif len(response.split("\n\n")) > 1 and re.search(rf"Option [{choices}]", response.split("\n\n")[-2].split("\n")[-1]):
                # Backoff strategy to find in CoT the most likely answer, when GPT-4 refuses to give a clear answer at the end.
                pred = re.search(rf"Option ([{choices}])", response.split("\n\n")[-2].split("\n")[-1]).group(1)
            elif re.search(rf"(?:Answer|ANSWER): (Option )?[{choices}]", response):
                pred = re.search(rf"(?:Answer|ANSWER): (Option )?([{choices}])", response).group(2)
            elif re.search(rf"[T|t]he .*?answer is\s*:?\s*[{choices}]", response):
                pred = re.search(rf"[T|t]he .*?answer is\s*:?\s*([{choices}])", response).group(1)
            elif re.search(rf"Answer:?\s*\*\*\s*[{choices}]\.?\s*.*?\*\*", response):
                pred = re.search(rf"Answer:?\s*\*\*\s*([{choices}])\.?\s*.*?\*\*", response).group(1)
            elif re.search(rf"best choice.*?would be:\s*[{choices}]", response):
                pred = re.search(rf"best choice.*?would be:\s*([{choices}])", response).group(1)
            elif re.search(rf"best choice.*?is:\s*[{choices}]", response):
                pred = re.search(rf"best choice.*?is:\s*([{choices}])", response).group(1)
            elif re.search(rf"Answer:?\s*[{choices}]", response):
                pred = re.search(rf"Answer:?\s*([{choices}])", response).group(1)
            elif re.search(rf"most likely diagnosis is:?\s*\*\*[{choices}]\.?\s*.*?\*\*", response):
                pred = re.search(rf"most likely diagnosis is:?\s*\*\*([{choices}])\.?\s*.*?\*\*", response).group(1)
            elif re.search(rf"most likely diagnosis is:?\s*[{choices}]", response):
                pred = re.search(rf"most likely diagnosis is:?\s*([{choices}])", response).group(1)
            elif (len(re.findall(rf"\*\*[{choices}]\.\s*.*?\*\*", response))) == 1 and re.search(rf"\*\*[{choices}]\.\s*.*?\*\*", response):
                pred = re.search(rf"\*\*([{choices}])\.\s*.*?\*\*", response).group(1)
            elif re.search(rf"most appropriate choice is:?\s*[{choices}]", response):
                pred = re.search(rf"most appropriate choice is:?\s*([{choices}])", response).group(1)
            elif re.search(rf"most relevant choice is:?\s*[{choices}]", response):
                pred = re.search(rf"most relevant choice is:?\s*([{choices}])", response).group(1)
            elif re.search(rf"[{choices}]\.", response_last_line):
                pred = re.search(rf"([{choices}])\.", response_last_line).group(1)
            elif re.search(rf"\*\*Answer\*\*:?\s*[{choices}]", response):
                pred = re.search(rf"\*\*Answer\*\*:?\s*([{choices}])", response).group(1)
            elif re.search(rf"\*\*Answer:\*\*\s*[{choices}]", response):
                pred = re.search(rf"\*\*Answer:\*\*\s*([{choices}])", response).group(1)
            elif re.search(rf"the best fit would be:?\s*[{choices}]", response):
                pred = re.search(rf"the best fit would be:?\s*([{choices}])", response).group(1)
            elif re.search(rf"the correct answer should be:?\s*\*\*[{choices}]", response):
                pred = re.search(rf"the correct answer should be:?\s*\*\*([{choices}])", response).group(1)
            elif re.search(rf"the correct answer should be:?\s*[{choices}]", response):
                pred = re.search(rf"the correct answer should be:?\s*([{choices}])", response).group(1)
            elif re.search(rf"([{choices}]) is the closest", response):
                pred = re.search(rf"([{choices}]) is the closest", response).group(1)
            else: # check last line
                # check if only one of the answer choices is in the last line of the response
                found1 = [re.search(f"\({choice}\)", response_last_line) is not None for choice in choices]
                found2 = [re.search(f"{choice}\.", response_last_line) is not None for choice in choices]
                found3 = [re.search(f" {choice} ", response_last_line) is not None for choice in choices]
                if sum(found1) == 1:
                    # this is the answer
                    pred = choices[found1.index(True)]
                elif sum(found2) == 1:
                    # this is the answer
                    pred = choices[found2.index(True)]
                elif sum(found3) == 1:
                    pred = choices[found3.index(True)]
        else:
            pred = response[0] # for direct answer, just take the first token
        assert str(pred) in choices, f"Model did not output one of {choices} as answer."
        return pred
