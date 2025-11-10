# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import yaml
import json
import random
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers.utils import logging


from trainer.grpo_trainer import InternVL3Trainer
from open_r1.trainer import GRPOConfig

logger = logging.get_logger(__name__)



# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False




def make_i2t_conversation(pos_cap, neg_cap):
            
    QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with the option's letter A or B."
    question = "Which caption best describe the given image?\nA.{}\nB.{}".format(pos_cap, neg_cap)
    return {
        "prompt": [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
                ],
            },
        ]
    }


def make_t2i_conversation(pos_cap):
    QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with First or Second."
    question = "Which image best match the below caption?\nCaption: {}".format(pos_cap)
    return {
        "prompt": [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
                ],
            },
        ]
    }

def make_matching_conversation(cap1):
    QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with Yes or No."
    question = "Does the below caption precisely describe the given image?\nCaption: {}".format(cap1)
    return {
        "prompt": [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)},
                ],
            },
        ]
    }



class LazySupervisedDatasetForCompositionalityV2(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args
        self.list_data_ids = []
        self.all_data_dict = {}
        
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                            self.all_data_dict.update(cur_data_dict)
                            all_data_id = list(cur_data_dict.keys())
                            all_data_id.sort()
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(all_data_id)
                        sampled_data_id = all_data_id[:sampling_number]
                        print(f"Loaded {len(sampled_data_id)} samples from {json_path}")
                        self.list_data_ids.extend(sampled_data_id) 
                    elif sampling_strategy == "all":
                        print(f"Load {len(all_data_id)} samples from {json_path}")
                        self.list_data_ids.extend(all_data_id)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        self.all_marks = [i%2 for i in range(len(self.list_data_ids))] 
        random.shuffle(self.all_marks)
        random.shuffle(self.list_data_ids) 
        print("total num: {}".format(len(self.list_data_ids)))

    def __len__(self):
        return len(self.list_data_ids)

    def __getitem__(self, i):
        did = self.list_data_ids[i]
        example = self.all_data_dict[did]
    
        image_root = self.script_args.image_root
        
        shard = example['shard']
        pos_image_path = os.path.join(image_root, shard, did+".image")
        neg_image_path = os.path.join(image_root, shard, did+".neg_image")
        
        pos_image = Image.open(pos_image_path).convert("RGB")
        neg_image = Image.open(neg_image_path).convert("RGB")

        pos_cap = example['pos cap']
        neg_cap = example['neg cap']

        mark = self.all_marks[i]


        
        if i%3 == 0:
            type = "image2text"
            ### 做image2text数据
            if mark == 1:
                prompt = make_i2t_conversation(pos_cap, neg_cap)['prompt'] 
                answer = "A"
            else:
                prompt = make_i2t_conversation(neg_cap, pos_cap)['prompt'] 
                answer = "B"
            prompt  = prompt[0]['content'][1]['text']
            prompt = "<image>/n"+prompt

            image = [pos_image]

        elif i%3 == 1:
            ### 做text2image数据
            type = "text2image"
            prompt = make_t2i_conversation(pos_cap)['prompt'][0]['content'][2]['text']
            prompt = "First: <image>/nSecond: <image>/n" + prompt
            if mark == 1:
                answer = "First"
                image = [pos_image, neg_image]
            else:
                answer = "Second"
                image = [neg_image, pos_image]
        else:
            type = "image_text_matching"
            if mark == 1:
                answer = "Yes"
                prompt = make_matching_conversation(pos_cap)['prompt']
            else:
                answer = "No"
                prompt = make_matching_conversation(neg_cap)['prompt']
            prompt = prompt[0]['content'][1]['text']
            prompt = "<image>/n"+prompt
            image = [pos_image]
        
    


        return {
            "type": type,
            "prompt": prompt,
            "solution": answer,
            "image": image,
        }



def multimodal_compositionality_reward_v2(completions, **kwargs):
    
    contents = [completion[0]["content"] for completion in completions]
    solutions = kwargs["solution"]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    rewards = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    types = kwargs['type']
    for cont, sol, ty in zip(contents, solutions, types):
        reward = 0.0
        content_answer_match = re.search(answer_tag_pattern, cont, re.DOTALL) 
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            if content_answer.lower() == sol.lower(): 
                reward = 1.0
        rewards.append(reward)
        

        ### 将输出结果写入文件
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Type: {ty}, Compostional reward: {reward} -------------\n")
                f.write(f"Content: {cont}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>\b(A|B|First|Second|Yes|No)\b</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

    


reward_funcs_registry = {
    "accuracy": multimodal_compositionality_reward_v2,
    "format": format_reward,
}




def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDatasetForCompositionalityV2(script_args.dataset_name, script_args)
    
    

    trainer_cls = InternVL3Trainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
