TASK=mmvp #cola, mmvp, winoground
REASON=False
device=0

model_path=OpenGVLab/InternVL3-2B
python eval/eval_cr_on_internvl3.py --model-path ${model_path} --task ${TASK} --device ${device} --reason ${REASON}