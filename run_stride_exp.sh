# run 5 times
for counter in {1..5}
do
    # test stride 1 to 5
    for stride in 1 3 4
    do
        mkdir -p "stats/stride/"$stride"stride"
        python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py \
         --batch_size 100 \
         --seed 0 \
         --num_layers 4 \
         --num_filters 64 \
         --dim_reduction_type strided_convolution \
	 --stride $stride \
         --experiment_name "stats/stride/"$stride"stride/"$counter"run" \
         --use_gpu True
    done
done
