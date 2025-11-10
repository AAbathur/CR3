# CR$^3$: Boosting Compositional Reasoning in MLLMs through Rule-based Reinforcement Learning
This is the official source code for paper "CR$^3$: Boosting Compositional Reasoning in MLLMs through Rule-based Reinforcement Learning".
This repository is based on the [VLM-R1](https://github.com/om-ai-lab/VLM-R1) and has been modified and extended according to our task requirements. We have added data processing and training code necessary for compositional reasoning, implemented the `Qwen2VLGRPOTrainerForCompositionality` and `InternVL3Trainer` class to support training Qwen2.5-VL and InternVL3 for compositional reasoning.




## Environment Preparation

only need to run the below commond
```
bash setup.sh
```

## Data processing
All experimental data in this repository are derived from TripletCLIP. The original dataset is available [here](https://huggingface.co/datasets/TripletCLIP/TripletCLIP-High-Quality). The filtered data used in our experiments can be found at: `train/data/tripletclip_shard_0-100_caption_final.json`.

Based on the TripletCLIP dataset, the image-text pairs are converted into three types of instruction data. The corresponding prompt templates are shown below.
| Task                                      | Prompt                                                                                                      |
|:------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| text-guided visual compositional reasoning  | First image: {image_1} Second image: {image_2} Which image best matches the caption below? Caption: {Caption_1} Output the final answer with First or Second.|
| visual-guided textual compositional reasoning |  {image} Which caption best describes the given image? A.{Caption_1} B.{Caption_2} Output the final answer with the option’s letter A or B.|
| compositional image-text matching | {image_1} Does the below caption precisely describe the given image? Caption: {Caption_1} Output the final answer with Yes or No. |

Noting: The processing code is already contained in "train/grpo_composition_internvl3.py" and "train/grpo_composition_qwen2_5_vl.py".

## Training
The experiments reported in the paper are conducted on 8*Nvidia A100 40GB GPUs.
The command used to train the model is:
```
bash run_scripts/train_grpo_qwen.sh ## train Qwen2.5-VL model
bash run_scripts/train_grpo_internvl3.sh ## train InternVL3 model
```

## Evaluation
The model's evaluation on in-domain benchmarks uses the below command:

```
# evaluate Qwen2.5-VL model
python eval/eval_cr_on_qwen.py --model-path /your/model/path --task task-name --device device_id --reason False
# evaluate InternVL3 model
python eval/eval_cr_on_internvl3.py  --model-path /your/model/path --task task-name --device device_id --reason False
```

The model's evaluation on out-of-domain benchmarks is based on the VLMEvalKit library. For evaluation code, please refer to the library's [ReadMe](https://github.com/open-compass/VLMEvalKit).


## License
This project is released under the [MIT License](LICENSE).
