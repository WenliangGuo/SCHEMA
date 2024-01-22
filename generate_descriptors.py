import os
import pandas as pd
import re
import os
import openai
import json
import argparse

openai_key = None #FILL IN YOUR OWN HERE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='crosstask', type=str, 
                    help='dataset')
parser.add_argument('--key', 
                    default='', type=str, 
                    help='OpenAI Key')
args = parser.parse_args()

openai.api_key = openai_key if openai_key is not None else args.key

def generate_prompts(task: str, step: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return [{"role": "user", "content": f"""
First, describe details of [step] for [goal] with one verb. 
Second, use 3 sentences to describe status changes of objects before and after [step],
avoiding using [word].

[goal]: Make Kimchi Fried Rice
[step]: add ham
[word]: add
Description:
Add diced ham into the fried rice
Before:
- The diced ham is separate from the pan.
- The pan contains fried rice.
- The pan has no ham on it.
After:
- The diced ham is mixed with the fried rice.
- The ham is on the pan.
- The pan contains ham.

[goal]: Make Pancakes
[step]: pour egg
[word]: pour
Description:
Pour egg into the pancake batter
Before:
- The egg is in a bowl.
- The pancake batter contains no egg.
- The egg is a whole.
After:
- The egg is mixed with the pancake batter.
- The egg is in the mixing bowl.
- The pancake batter contains egg.

[goal]: {task}
[step]: {step}
[word]: {step.split(' ')[0]}
"""}]


def create_chatgpt_responses(prompt):
    while True:
        try:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                messages=prompt,
                                temperature=0,
                                max_tokens=300,
                                presence_penalty=0
                                )
            break
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
    return response


def stringtolist(description):
    outputs = {}
    # outputs["description"] = description.split("Description:\n")[1].split("\nBefore:")[0]
    outputs["before"] = [d[2:] for d in description.split("Before:\n")[1].split("After:\n")[0].split('\n') if (d != '') and (d.startswith('- '))]
    outputs["after"] = [d[2:] for d in description.split("After:\n")[1].split('\n') if (d != '') and (d.startswith('- '))]
    return outputs


if __name__ == "__main__":
    ## Load data
    if args.dataset == "crosstask":
        from utils import read_task_info
        root = 'dataset/crosstask'
        primary_info = read_task_info(os.path.join(root, 'crosstask_release', 'tasks_primary.txt'))
        task_steps = [(primary_info['title'][task_idx], step) for task_idx in primary_info['title'] for step in primary_info['steps'][task_idx]]

    elif args.dataset == "coin":
        root_path = "dataset/coin"
        tax_path = os.path.join(root_path, 'taxonomy.xlsx')
        dt_df = pd.read_excel(tax_path, sheet_name='target_action_mapping')
        ## create a dictionary based on the taxonomy
        dt_dict = []
        for i in range(len(dt_df)):
            task_id = dt_df['Target Id'][i]
            taeget_label = re.sub(r'([A-Z])', r' \1', dt_df['Target Label'][i]).strip()
            action_id = dt_df['Action Id'][i]
            action_label = dt_df['Action Label'][i]
            dt_dict.append({'task_id': task_id, 'target_label': taeget_label, 'action_id': action_id, 'action_label': action_label})
        task_steps = [(d['target_label'], d['action_label']) for d in dt_dict]

    elif args.dataset == "niv":
        with open("dataset/niv/niv_task.json", 'r') as f:
            niv_info = json.load(f)
        task_steps = []
        for task, steps in niv_info.items():
            for step in steps:
                task_steps.append((task, step))

    ## Generate prompts
    prompts_chatgpt = [generate_prompts(task, step) for (task, step) in task_steps]

    ## Get GPT's response
    responses = []
    for i, prompt in enumerate(prompts_chatgpt):
        responses.append(create_chatgpt_responses(prompt))
    response_texts = [r['message']['content'] for resp in responses for r in resp['choices']]

    ## Parse response to get descriptors
    descriptions = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {}
    for (task, step), desc in zip(task_steps, descriptions):
        if task not in descriptors:
            descriptors[task] = {}
        descriptors[task][step] = desc

    ## Save descriptors
    with open(os.path.join("data", f"descriptors_{args.dataset}.json"), 'w') as f:
        json.dump(descriptors, f)