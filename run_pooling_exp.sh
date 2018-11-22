# run 5 times
for counter in {1..5}
do
    # test 4 dim reduction
    for dim_reduction_type in max_pooling avg_pooling
    do
	mkdir -p "stats/pooling/"$dim_reduction_type
	python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py \
	       --batch_size 100 \
	       --seed 0 \
	       --num_layers 4 \
	       --num_filters 64 \
	       --dim_reduction_type $dim_reduction_type \
	       --experiment_name "stats/pooling/"$dim_reduction_type"/"$counter"run" \
	       --use_gpu True
    done
done
