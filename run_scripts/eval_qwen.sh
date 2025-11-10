TASK=vsr #cola, mmvp, winoground
#REASON=False
device=0

model_path=Qwen/Qwen2.5-VL-3B-Instruct
python eval/eval_cr_on_qwen.py --model-path ${model_path} --task ${TASK} --device ${device} --reason ${REASON}
