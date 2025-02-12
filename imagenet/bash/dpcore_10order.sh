for i in 0 1 2 3 4 5 6 7 8 9
do
    python imagenetc_order.py --cfg ./cfgs/dpcore_10_random_order/dpcore$i.yaml  --data_dir $DATA_DIR
done


