# Dataset class for Bias Benchmark QA (BBQ) dataset

from my_datasets.dataset import Dataset


class BBQDataset(Dataset):
    def __init__(self, name, dataset_path):
        super().__init__(name, dataset_path)
        self.direct_answer_trigger = """Please try to be as accurate as possible and avoid answering unknown when there is enough information to answer.\n\n"""

    def format_prompt_basic(self, idx, context_idx=0, double_space=True, context_ans=False):
        """
        Formats a single BBQ question for the LLM in most basic format (without CoT, few shot examples, etc.).
        Args:
            idx: index of the question
            context_idx: index of the context to use (weak evidence 0 or 1)
            context_ans: whether the answer changes depending on the context
        Returns:
            prompt: a formatted prompt for the LLM
        """
        sep = "\n\n"
        if not double_space:
            sep = "\n"
        row = self.data[idx]
        evidence = row["weak_evidence"][context_idx]
        if context_ans:
            prompt = f"""{row["context"]} {evidence}{sep}{row["question"]}{sep}Answer choices:{sep}(A) {row["ans0"][context_idx]}{sep}(B) {row["ans1"][context_idx]}{sep}(C) {row["ans2"][context_idx]}"""
        else:
            prompt = f"""{row["context"]} {evidence}{sep}{row["question"]}{sep}Answer choices:{sep}(A) {row["ans0"]}{sep}(B) {row["ans1"]}{sep}(C) {row["ans2"]}"""
        return prompt
    
    def format_question_info(self, idx, context_idx=0, context_ans=False):
        row = self.data[idx]
        evidence = row["weak_evidence"][context_idx]
        question_info = f"Context: {row['context']} {evidence}\n"
        question_info += f"Question: {row['question']}\n"
        if context_ans:
            question_info += f"Answer choices:\n(A) {row['ans0'][context_idx]}\n(B) {row['ans1'][context_idx]}\n(C) {row['ans2'][context_idx]}\n"
        else:
            question_info += f"Answer choices:\n(A) {row['ans0']}\n(B) {row['ans1']}\n(C) {row['ans2']}\n"
        return question_info
    
    def parse_counterfactual_output(self, counterfactual_output, includes_quality_checks=False):
        """
        Parses and validates the counterfactual output by checking that all expected entries are present and extracting them.
        Args:
            counterfactual_output: str, the counterfactual output to parse
            includes_quality_checks: bool, whether the outputs include quality check information
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
        assert counterfactual_output_lines[line_idx].startswith("(A)"), f"Edited Answer Choice (A) not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans0"] = counterfactual_output_lines[line_idx].split("(A)")[1].strip()
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("(B)"), f"Edited Answer Choice (B) not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans1"] = counterfactual_output_lines[line_idx].split("(B)")[1].strip()
        line_idx += 1
        assert counterfactual_output_lines[line_idx].startswith("(C)"), f"Edited Answer Choice (C) not where expected. Counterfactual output: {counterfactual_output}"
        output_dict["edited_ans2"] = counterfactual_output_lines[line_idx].split("(C)")[1].strip()
        if includes_quality_checks:
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
            line_idx += 1
            if not counterfactual_output_lines[line_idx]:
                line_idx += 1
            assert counterfactual_output_lines[line_idx].startswith("Comments on edit success:"), f"Comments on edit success not where expected. Counterfactual output: {counterfactual_output}"
            output_dict["success_comments"] = counterfactual_output_lines[line_idx].split("Comments on edit success:")[1].strip()
            line_idx += 1
            if not counterfactual_output_lines[line_idx]:
                line_idx += 1
            assert counterfactual_output_lines[line_idx].startswith("Successful YES/NO:"), f"Edit success decision not where expected. Counterfactual output: {counterfactual_output}"
            assert counterfactual_output_lines[line_idx].split(":")[1].strip() in ["YES", "NO"], f"Edit success decision not YES or NO. Counterfactual output: {counterfactual_output}"
            output_dict["success"] = counterfactual_output_lines[line_idx].split(":")[1].strip()
        return output_dict
