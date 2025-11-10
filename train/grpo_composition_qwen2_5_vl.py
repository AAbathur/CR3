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
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers.utils import logging
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func


from open_r1.trainer import GRPOConfig
from trainer.grpo_trainer import Qwen2VLGRPOTrainerForCompositionality

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


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
    stage: Optional[int] = field(
        default=None,
        metadata={"help": "stage of train"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


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


def make_vqa_conversation(ques):
    QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Must answer the question with short words or phrases."
    question = ques
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





class LazySupervisedDatasetForCompositionality(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments,):
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
            typex = "image2text"
            if mark == 1:
                prompt = make_i2t_conversation(pos_cap, neg_cap)['prompt'] 
                answer = "A"
            else:
                prompt = make_i2t_conversation(neg_cap, pos_cap)['prompt'] 
                answer = "B"
            image = [pos_image]

        elif i%3 == 1:
            typex = "text2image"
            prompt = make_t2i_conversation(pos_cap)['prompt']
            if mark == 1:
                answer = "First"
                image = [pos_image, neg_image]
            else:
                answer = "Second"
                image = [neg_image, pos_image]
        else:
            typex = "image_text_matching"
            if mark == 1:
                answer = "Yes"
                prompt = make_matching_conversation(pos_cap)['prompt']
            else:
                answer = "No"
                prompt = make_matching_conversation(neg_cap)['prompt']
            image = [pos_image]
        
           
        return {
            "type": typex,
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
    type2count = {}
    for cont, sol, ty in zip(contents, solutions, types):
        reward = 0.0
        content_answer_match = re.search(answer_tag_pattern, cont, re.DOTALL) 
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip() 
            if content_answer.lower() == sol.lower(): 
                reward = 1.0
            if content_answer == sol: 
                reward = 1.0
        
        type2count[ty] = type2count.get(ty, [0,0])
        type2count[ty][1] += 1
        if reward != 0:
            type2count[ty][0] += 1

        rewards.append(reward)
        

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Type: {ty}, Compostional reward: {reward} -------------\n")
                f.write(f"Content: {cont}\n")
                f.write(f"Solution: {sol}\n")
    return rewards, type2count

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>\b(A|B|First|Second|Yes|No)\b</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches], None
    


reward_funcs_registry = {
    "accuracy": multimodal_compositionality_reward_v2,
    "format": format_reward,
}




def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = LazySupervisedDatasetForCompositionality(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainerForCompositionality
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
