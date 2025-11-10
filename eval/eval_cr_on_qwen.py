
import os
import re
import json
import argparse
from tqdm import tqdm

from sklearn.utils import shuffle
import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--task", choices=['winoground','mmvp','cola', 'vsr'], )
parser.add_argument("--device", type=int)
parser.add_argument("--reason", type=str)
args = parser.parse_args()





def load_model(MODEL_PATH, device_id):
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        #attn_implementation="sdpa",
        device_map="cuda:{}".format(device_id),
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def get_image2text_prompt(image_path, cap1, cap2, reasoning):
    if reasoning:
        #QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with the option's letter A or B."
        QUESTION_TEMPLATE = "{Question}\nMust first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. The final result contains only the option letter A or B."
        question = "Which caption best describe the given image?\nA.{}\nB.{}".format(cap1, cap2)
    else:
        ## W/O reasoning, directly prompt
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with the option's letter A or B."
        question = "Which caption best describe the given image?\nA. {}\nB. {}".format(cap1, cap2)
    
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message


def get_image2text_prompt_v2(cap1, cap2, reasoning):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with the option's letter A or B."
        question = "Which caption best describe the given image?\nA.{}\nB.{}".format(cap1, cap2)
    else:
        ## W/O reasoning, directly prompt
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with the option's letter A or B."
        question = "Which caption best describe the given image?\nA.{}\nB.{}".format(cap1, cap2)
    
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message

def get_text2image_prompt(image_path1, image_path2, cap1, reasoning):
    ## v1
    #QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    #question = "Which image best match the below caption?\nCaption:{}\nJust output first or second.".format(cap1)
    ## v2
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with First or Second."
        question = "Which image best match the below caption?\nCaption:{}".format(cap1)
        ### question = "Describe the contents of the images sequentially, with different image descriptions separated by the # symbol."
    ## W/O reasoning, directly prompt
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with First or Second."
        question = "Which image best match the below caption?\nCaption:{}".format(cap1)
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path1}"
            },
            {
                "type": "image", 
                "image": f"file://{image_path2}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message


def get_itm_prompt(image_path1, text1, reasoning):
    ## v1
    #QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    #question = "Which image best match the below caption?\nCaption:{}\nJust output first or second.".format(cap1)
    ## v2
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with Yes or No."
        question = "Does the below caption precisely describe the given image?\nCaption: {}".format(text1)
    ## W/O reasoning, directly prompt
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with Yes or No."
        question = "Does the below caption precisely describe the given image?\nCaption: {}".format(text1)
    message = [
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path1}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message


def get_caption_prompt(image_path1, reasoning):
    ## v1
    #QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    #question = "Which image best match the below caption?\nCaption:{}\nJust output first or second.".format(cap1)
    ## v2
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with Yes or No."
        question = "Describe the given image in detail"
    ## W/O reasoning, directly prompt
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with Yes or No."
        question = "Describe the given image in detail"
    message = [
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path1}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message

def get_2img_cap_prompt(image_path1, image_path2, reasoning):
    ## v1
    #QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    #question = "Which image best match the below caption?\nCaption:{}\nJust output first or second.".format(cap1)
    ## v2
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with First or Second."
        question = "Describe the contents of the images sequentially, with different image descriptions separated by the # symbol."
    ## W/O reasoning, directly prompt
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with First or Second."
        question = "Describe the contents of the images sequentially, with different image descriptions separated by the # symbol."
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path1}"
            },
            {
                "type": "image", 
                "image": f"file://{image_path2}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message

def process_for_input(processor, batch_messages, device_id): 
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:{}".format(device_id))
    inputs = inputs.to(torch.bfloat16)
    return inputs





def inference_and_decode(model, processor, batch_inputs):
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs, use_cache=True, max_new_tokens=512, do_sample=True, temperature=1.0,)
            
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text

def extract_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        return content_answer
    return None





########################################################################################################################
### Winoground
########################################################################################################################


def load_winoground():
    inpath = "/your/local/data/path/Winoground-data/data_examples.jsonl"
    img_path_temp = "/your/local/data/path/Winoground-data/images/{}.png"
    with open(inpath, "r", encoding="utf-8") as f1:
        for i, line in enumerate(f1):
            cont = json.loads(line.rstrip())
            cap0 = cont['caption_0']
            cap1 = cont['caption_1']
            img0_path = img_path_temp.format(cont['image_0'])
            img1_path = img_path_temp.format(cont['image_1'])
            yield cap0, cap1, img0_path, img1_path 

def process_winoground_for_qwen2vl(pos_first=True, REASON=True):
    wino_ds = load_winoground()
    all_sample = []
    for data in tqdm(wino_ds):
        cap0, cap1, img0_path, img1_path = data
        #### pos-first
        if pos_first:
            img0_cap = get_image2text_prompt(img0_path, cap0, cap1, reasoning=REASON)
            img1_cap = get_image2text_prompt(img1_path, cap1, cap0, reasoning=REASON)
            cap0_img = get_text2image_prompt(img0_path, img1_path, cap0, reasoning=REASON)
            cap1_img = get_text2image_prompt(img1_path, img0_path, cap1, reasoning=REASON)
        else: ## neg-first
            img0_cap = get_image2text_prompt(img0_path, cap1, cap0, reasoning=REASON)
            img1_cap = get_image2text_prompt(img1_path, cap0, cap1, reasoning=REASON)
            cap0_img = get_text2image_prompt(img1_path, img0_path, cap0, reasoning=REASON)
            cap1_img = get_text2image_prompt(img0_path, img1_path, cap1, reasoning=REASON)
        all_sample.append([img0_cap, img1_cap, cap0_img, cap1_img])
    return all_sample



def accuracy_on_winoground(all_preds, pos_first, extract_ans=True):
    nbad, _, total = 0, 0, 0
    if pos_first:
        i2t_label = "A"
        t2i_label = "first"
    else:
        i2t_label = "B"
        t2i_label = "second"
    nbad = 0
    i2t_count, t2i_count, group_count = 0, 0, 0
    for img0_i2t, img1_i2t, cap0_t2i, cap1_t2i in all_preds:
        total += 1
        if extract_ans:
            img0_i2t_ans = extract_answer(img0_i2t)
            img1_i2t_ans = extract_answer(img1_i2t)
            cap0_t2i_ans = extract_answer(cap0_t2i)
            cap1_t2i_ans = extract_answer(cap1_t2i)
        else:
            img0_i2t_ans = img0_i2t
            img1_i2t_ans = img1_i2t
            cap0_t2i_ans = cap0_t2i
            cap1_t2i_ans = cap1_t2i
        
        print(total, repr(img0_i2t_ans), repr(img1_i2t_ans), repr(cap0_t2i_ans), repr(cap1_t2i_ans))

        if (not img0_i2t_ans) or (not img1_i2t_ans) or (not cap0_t2i_ans) or (not cap1_t2i_ans):
            nbad += 1
            continue

        flag0, flag1 = False, False
        if img0_i2t_ans == img1_i2t_ans == i2t_label:
            i2t_count += 1
            flag0 = True
        if cap0_t2i_ans.lower() == cap1_t2i_ans.lower() == t2i_label:
            t2i_count += 1
            flag1 = True
        print("\t image2text: {}, text2image: {}".format(flag0, flag1))
        if flag0 and flag1:
            group_count += 1
    
    return nbad, i2t_count, t2i_count, group_count, total


def process_winoground_for_qwen2vl_v2(ds, REASON=True):
    if ds == "winoground":
        all_data = load_winoground()
        nhalf = 200
    elif ds == "cola":
        all_data = load_cola()
        nhalf = len(all_data) // 2

    all_sample = []
    all_label = []
    
    ### 这样处理的好处是，可以确保正负样本数量相同，评估时结果一致
    all_i2t_mark = [0 for _ in range(nhalf)] + [1 for _ in range(nhalf)]
    all_i2t_mark = shuffle(all_i2t_mark, random_state=1123)
    all_t2i_mark = [1 for _ in range(nhalf)] + [0 for _ in range(nhalf)]
    all_t2i_mark = shuffle(all_t2i_mark, random_state=1234)

    all_new_data = []
    for i, data in enumerate(all_data):
        cap0, cap1, img0_path, img1_path = data
        all_new_data.append([cap0, cap1, img0_path, img1_path])

        i2t_mark = all_i2t_mark[i]
        t2i_mark = all_t2i_mark[i]
        #### pos-first
        if i2t_mark == 1:
            img0_cap = get_image2text_prompt(img0_path, cap0, cap1, reasoning=REASON)
            img1_cap = get_image2text_prompt(img1_path, cap0, cap1, reasoning=REASON)
            label_i0 = "A"
            label_i1 = "B"
        else:
            img0_cap = get_image2text_prompt(img0_path, cap1, cap0, reasoning=REASON)
            img1_cap = get_image2text_prompt(img1_path, cap1, cap0, reasoning=REASON)
            label_i0 = "B"
            label_i1 = "A"

        if t2i_mark == 1:
            cap0_img = get_text2image_prompt(img0_path, img1_path, cap0, reasoning=REASON)
            cap1_img = get_text2image_prompt(img0_path, img1_path, cap1, reasoning=REASON)
            label_c0 = "first"
            label_c1 = "second"
        else:
            cap0_img = get_text2image_prompt(img1_path, img0_path, cap0, reasoning=REASON)
            cap1_img = get_text2image_prompt(img1_path, img0_path, cap1, reasoning=REASON)
            label_c0 = "second"
            label_c1 = "first"
        all_sample.append([img0_cap, img1_cap, cap0_img, cap1_img])
        all_label.append([label_i0, label_i1, label_c0, label_c1])
    return all_sample, all_label, all_new_data



def accuracy_on_winoground_v2(all_preds, all_label, all_sample, extract_ans=True):
    bad_i2t = 0
    nbad, total = 0, 0
    i2t_count, t2i_count, group_count = 0, 0, 0
    for i, (img0_i2t, img1_i2t, cap0_t2i, cap1_t2i) in enumerate(all_preds):
        q1, q2, q3, q4 = all_sample[i]
        img0_i2t_label, img1_i2t_label, cap0_t2i_label, cap1_t2i_label = all_label[i]
        if extract_ans:
            img0_i2t_ans = extract_answer(img0_i2t)
            img1_i2t_ans = extract_answer(img1_i2t)
            cap0_t2i_ans = extract_answer(cap0_t2i)
            cap1_t2i_ans = extract_answer(cap1_t2i)
        else:
            if "A" in img0_i2t:
                img0_i2t_ans = "A"
            elif "B" in img0_i2t:
                img0_i2t_ans = "B"
            else:
                img0_i2t_ans = img0_i2t
            
            if "A" in img1_i2t:
                img1_i2t_ans = "A"
            elif "B" in img1_i2t:
                img1_i2t_ans = "B"
            else:
                img1_i2t_ans = img1_i2t 
            cap0_t2i_ans = cap0_t2i
            cap1_t2i_ans = cap1_t2i

        p1, p2 = False, False
        if img0_i2t_ans not in ["A", "B"]:
            p1 = True
            bad_i2t += 1
        if img1_i2t_ans not in ["A", "B"]:
            p2 = True
            bad_i2t += 1
        if p1 or p2:
            continue

        total += 1
        if (not img0_i2t_ans) or (not img1_i2t_ans) or (not cap0_t2i_ans) or (not cap1_t2i_ans):
            nbad += 1
            continue
        flag0, flag1 = False, False
        if (img0_i2t_ans == img0_i2t_label) and (img1_i2t_ans == img1_i2t_label):
            i2t_count += 1
            flag0 = True
        if (cap0_t2i_ans.lower() == cap0_t2i_label) and (cap1_t2i_ans.lower() == cap1_t2i_label):
            t2i_count += 1
            flag1 = True
        if flag0 and flag1:
            group_count += 1
    
    print("bad pred answer num: ", bad_i2t, "total num", total*2)
    return nbad, i2t_count, t2i_count, group_count, total



def evaluate_on_winoground_v2(ds, model_path, device_id, REASON, half=False):
    BSZ = 1
    model, processor = load_model(model_path, device_id)
    all_sample, all_label, all_data = process_winoground_for_qwen2vl_v2(ds, REASON=REASON)


    all_outputs = []
    for i in tqdm(range(0, len(all_sample), BSZ)):
        batch_sample = []
        for sample in all_sample[i:i+BSZ]:
            batch_sample += sample 
        assert len(batch_sample) == 4
        batch_inputs = process_for_input(processor, batch_sample, device_id)
        batch_inputs = batch_inputs.to(torch.bfloat16)
        batch_output_text = inference_and_decode(model, processor, batch_inputs)
        for j in range(0, len(batch_sample), 4):
            all_outputs.append(batch_output_text[j:j+4])

    
    nbad, i2t_count, t2i_count, group_count, total = accuracy_on_winoground_v2(all_outputs, all_label, all_sample, extract_ans=REASON)
    print("model_path", model_path)
    print("data: {}, REASON: {}".format(ds, REASON))
    print("nbad: {}, total: {}, bad_ratio: {:.4f}".format(nbad, total, nbad/total))
    print("image2text_count: {}, total: {}, image2text_acc: {:.4f}".format(i2t_count, total, i2t_count/total))
    print("text2image_count: {}, total: {}, text2image_acc: {:.4f}".format(t2i_count, total, t2i_count/total))
    print("group_count: {}, total: {}, group_acc: {:.4f}".format(group_count, total, group_count/total))
    


def load_cola():
    def get_img_path(img_url, img_dir):
        img_name = img_url.split("/")[-1]
        return os.path.join(img_dir, img_name)
    path1 = "/your/local/data/path/Cola/COLA_multiobjects_matching_benchmark.json"
    img_dir = "/your/local/data/path/Cola/cola-multi-obj-images"
    
    all_data = []
    with open(path1, "r", encoding="utf-8") as f1:
        cont = json.load(f1)
        for i,ct in enumerate(cont):
            #if i>3: break
            url0, cap0, url1, cap1 = ct
            img0_path = get_img_path(url0, img_dir)
            img1_path = get_img_path(url1, img_dir)
            all_data.append([cap0, cap1, img0_path, img1_path])
            
    return all_data



########################################################################################################################
### MMVP
########################################################################################################################


def get_mmvp_prompt(image_path, ques, option1, option2, reasoning=True):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with the option's letter A or B."
        question = "{}\nA.{}\nB.{}".format(ques, option1, option2)
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with the option's letter A or B."
        question = "{}\nA.{}\nB.{}".format(ques, option1, option2)
    
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{image_path}"
                #"image": image_path
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=question)
            }
        ]
    }]
    return message

def process_mmvp_for_qwen2vl(reasoning):
    dir1 = "/your/local/data/path/MMVP"
    ques_file = os.path.join(dir1, "Questions.csv")
    img_dir = os.path.join(dir1, "MMVP_Images")

    df = pd.read_csv(ques_file)
    all_prompt, all_label = [], []
    for idx, row in tqdm(df.iterrows(), total=300):
        answer = "A" if row["Correct Answer"] == "(a)" else "B"
        all_label.append(answer)
        option1, option2 = row["Options"].split(" (b) ")
        option1 = option1.split("(a) ")[1]
        ques = row['Question']
        img_path = os.path.join(img_dir, f"{idx+1}.jpg")
        prompt1 = get_mmvp_prompt(img_path, ques=ques, option1=option1, option2=option2, reasoning=reasoning)
        all_prompt.append(prompt1)
    return all_prompt, all_label


def accuracy_on_mmvp(all_pred, all_label, extract_ans=True):
    nbad, count, total = 0, 0, 0
    all_mark = []
    for content, label in zip(all_pred, all_label):
        total += 1
        if extract_ans:
            cont_ans = extract_answer(content)
        else:
            cont_ans = content
        if not cont_ans:
            nbad += 1
            all_mark.append(0)
        elif cont_ans == label:
            count += 1
            all_mark.append(1)
        else:
            all_mark.append(0)
    print("MMVP, nbad: {}, total: {}, count: {}, Single ACC: {:.4f}".format(nbad, total, count, count/total))
    pair_count, pair_total = 0, 0
    for i in range(0, len(all_mark), 2):
        pair_total += 1
        mark1 = all_mark[i]
        mark2 = all_mark[i+1]
        if mark1 == mark2 == 1:
            pair_count += 1
    print("pair count: {}, pair_total: {} Pair Acc: {:.4f}".format(pair_count, pair_total, pair_count/pair_total))



def evaluate_on_mmvp(model_path, device_id, REASON):
    BSZ=1
    model, processor = load_model(model_path, device_id)

    all_sample, all_label = process_mmvp_for_qwen2vl(reasoning=REASON)


    all_outputs = []
    for i in tqdm(range(0, len(all_sample), BSZ)):
        batch_sample = all_sample[i:i + BSZ]
        batch_inputs = process_for_input(processor, batch_sample, device_id)
        batch_output_text = inference_and_decode(model, processor, batch_inputs)
        if i<10:
            print(i, all_sample[i])
            print("output", batch_output_text[0])
        all_outputs.extend(batch_output_text)
    
    accuracy_on_mmvp(all_outputs, all_label, extract_ans=REASON)
    print("Reason", REASON, "model_path", model_path)






########################################################################################################################
### VSR
########################################################################################################################


def process_vsr_for_qwen2vl(reasoning, split):

    if split == "dev":
        path1 = "/your/local/data/path/VSR-random/dev.jsonl"  
    elif split == "test":
        path1 = "/your/local/data/path/VSR-random/test.jsonl" 
    img_dir = "/your/local/coco_2017/data_path/coco/train2017"
    
    total, count = 0, 0
    all_prompt, all_label = [], []
    all_img_path, all_caps = [], []
    with open(path1, "r", encoding="utf-8") as f1:
        for i, line in enumerate(f1):
            #if i>10: break
            total += 1
            cont = json.loads(line.rstrip())
            img_path = os.path.join(img_dir, cont['image'])
            img_link = cont['image_link']
            if "val2017" in img_link:
                count += 1
                continue
            cap = cont['caption']
            label = cont['label']
            if label == 1:
                answer = 'yes'
            else:
                answer = 'no'
            prompt = get_itm_prompt(img_path, cap, reasoning)
            all_prompt.append(prompt)
            all_label.append(answer)
            all_img_path.append(img_path)
            all_caps.append(cap)
    print("total: {}, abandon num: {}".format(total, count))
    return all_prompt, all_label, all_img_path, all_caps


def accuracy_on_vsr(all_pred, all_label, extract_ans=True):
    nbad, count, total = 0, 0, 0
    all_mark = []
    all_clean_pred = []
    for content, label in zip(all_pred, all_label):
        total += 1
        if extract_ans:
            cont_ans = extract_answer(content)
            print("total", total)
            print("raw ans", repr(content))
            print("extract ans", cont_ans)
        else:
            cont_ans = content
        cont_ans = cont_ans.replace(".", "")
        all_clean_pred.append(cont_ans.lower())
        if not cont_ans:
            nbad += 1
            all_mark.append(0)
        elif cont_ans.lower() == label:
            count += 1
            all_mark.append(1)
        else:
            all_mark.append(0)
    print("VSR, nbad: {}, total: {}, count: {}, Single ACC: {:.4f}".format(nbad, total, count, count/total))
    return all_clean_pred

def evaluate_on_vsr(model_path, device_id, REASON):
    split = "test"
    model, processor = load_model(model_path, device_id)
    all_sample, all_label, all_img_path, all_caps = process_vsr_for_qwen2vl(reasoning=REASON, split="test")

    BSZ=8
    all_outputs = []
    for i in tqdm(range(0, len(all_sample), BSZ)):
        #if i>10: break
        batch_sample = all_sample[i:i + BSZ]
        batch_inputs = process_for_input(processor, batch_sample, device_id)
        batch_output_text = inference_and_decode(model, processor, batch_inputs)
        if i<10:
            print(i, all_sample[i])
            print("output", batch_output_text[0])
        all_outputs.extend(batch_output_text)
    
    accuracy_on_vsr(all_outputs, all_label, extract_ans=REASON)
    print("Split:", split, "REASON:", REASON, "model_path:", model_path)
    



def main():
    if args.reason == "False":
        xreason = False
    elif args.reason == "True":
        xreason = True
    print("Xreason", xreason)
    print("args", args)

    if args.task == "winoground":
        evaluate_on_winoground_v2('winoground', args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "cola":
        evaluate_on_winoground_v2('cola', args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "mmvp":
        evaluate_on_mmvp(args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "vsr":
        evaluate_on_vsr(args.model_path, device_id=args.device, REASON=xreason)
    
    print("Xreason", xreason)
    print("args", args)




if __name__ == "__main__":
    
    main()
    
