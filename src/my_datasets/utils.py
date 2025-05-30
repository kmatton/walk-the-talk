# dataset utility functions

import random
import re


####################################################################################################
## BBQ data Utils ##
####################################################################################################

def parse_question_from_prompt_bbq(prompt):
    """
    Parses a question from a BBQ prompt.
    Args:
        prompt: BBQ prompt
    Returns:
        question: the question
    """
    question = prompt.split("\n\n")[1]
    return question


####################################################################################################
## MEDQA data Utils ##
####################################################################################################

def get_options_in_str_medqa(example, alt=False):
    options = []
    for k in sorted(example["answer_choices"].keys()):
        # Some options has newline.
        if alt:
            op = example["answer_choices"][k].strip(' \n')
        else:
            op = example["answer_choices"][k].replace('\n"', "")
        options.append(f"{k}. {op}")
    options = "\n".join(options)
    return options


def extract_context_and_final_question(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +|\n+', text)
    # Identify the last sentence
    final_sentence = sentences[-1]
    # Check if the final sentence is a question
    assert final_sentence.strip().endswith('?'), "final sentence is not a question"
    context = ' '.join(sentences[:-1]).strip()
    question = final_sentence.strip()
    return context, question


# prompting strategies

def parse_explanation(example):
    cot, answer_rank = example["explanation"].rsplit("\n", 1)
    match = re.search(r"Answer: \[([A-D])\] > \[([A-D])\] > \[([A-D])\] > \[([A-D])\].", answer_rank)
    answers = []
    try:
        for i in range(1, 5):
            answers.append(match.group(i))
    except:
        breakpoint()
    cot = cot.rsplit("\n\n## ", 1)[0]
    assert f"[{answers[0]}]" in cot, breakpoint()
    return cot, answers

def knn_few_shot_rank_cot_md(example, **kwargs):
    question = example["question"]
    options = get_options_in_str_medqa(example, alt=True)
    prompt = ""
    choices_list = [("A", "BCD"), ("B", "CDA"), ("C", "DAB"), ("D", "ABC")]
    choices_list = choices_list + [random.choice(choices_list)]
    choices_list = random.sample(choices_list, 5)
    for ex, choices in zip(example["few_shot"][::-1], choices_list):
        assert ex["explanation"].endswith("]."), breakpoint()

        w, xyz = choices
        op_order = [w] + random.sample(xyz, 3)
        cot, ans_rank = parse_explanation(ex)
        op_map = {op2: op1 for op1, op2 in zip(ans_rank, op_order)}
        answer_choices = ex["answer_choices"]
        ex["answer_choices"] = {k: answer_choices[op_map[k]] for k in sorted(op_map)}
        answers = "Answer: " + " > ".join([f"[{a}]" for a in op_order]) + "."
        cot = cot.replace(f"[{ans_rank[0]}]", f"[{w}]")
        ex['explanation'] = f"{cot}\n\n## List all options from most likely to least likely\n{answers}"

        ex_options = get_options_in_str_medqa(ex, alt=True)
        prompt += f"## Question\n{ex['question']}\n\n{ex_options}\n\n## Answer\n{ex['explanation']}\n\n"
    prompt += f"## Question\n{question}\n\n{options}\n\n## Answer\n"
    return prompt


def few_shot_rank_cot_knn_md(example, **kwargs):
    question = example["question"]
    options = get_options_in_str_medqa(example)

    prompt = "You are tasked with composing comprehensive solutions for the Medical Licensing Examination. The objective is to devise these solutions in a manner that facilitates student learning, not just by providing the correct answers, but also by teaching how to systematically approach, analyze, and solve the problem at hand. The following set of questions represents the kind of challenges you are aiming to elucidate in your solution sets.\n\n"
    for ex in example["few_shot"]:
        ex_options = get_options_in_str_medqa(ex)
        prompt += (
            f"## Question\n{ex['question']}\n\n{ex_options}\n\n"
            f"## Answer\n{ex['explanation']}\n\n"
        )
    prompt += (
        f"## Question\n{question}\n\n{options}\n\n"
        "## Answer\n"
    )

    return prompt
