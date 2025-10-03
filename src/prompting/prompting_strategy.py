# class for specifying the prompting strategy


class PromptingStrategy:
    def __init__(self, cot, few_shot, knn_rank, few_shot_prompt_name=None, add_instr=None):
        """
        Class for specifying the prompting strategy.
        Args:
            cot: whether to use CoT or direct answer trigger
            few_shot: whether to add few-shot examples to prompt
            knn_rank: whether to use knn rank (for now, only applicable to MedQA)
            few_shot_prompt_name: name of few shot prompt to use
            add_instr: additional instructions to add to prompt
        """
        self.cot = cot 
        self.few_shot = few_shot
        self.knn_rank = knn_rank
        if self.few_shot:
            assert few_shot_prompt_name is not None, "few_shot_prompt_name must be specified if few_shot is True"
        self.few_shot_prompt_name = few_shot_prompt_name
        self.add_instr = add_instr
