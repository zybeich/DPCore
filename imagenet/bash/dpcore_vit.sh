# export PYTHONPATH=
# conda deactivate
# conda activate vida
data_dir=$DATA_DIR

python imagenetc.py --cfg ./cfgs/vit/dpcore.yaml  --data_dir $data_dir


