tfl_config_path=$1
ckpt_path=$2
eval_path=data/highlight_single_inference.jsonl
echo ${ckpt_path}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python FlashVTG/single_inference.py \
${tfl_config_path} \
--resume ${ckpt_path} \
--eval_path ${eval_path} \
${@:4}
