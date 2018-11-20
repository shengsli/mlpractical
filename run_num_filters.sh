# run 5 times
for counter in {1..5}
do
    # test 4 dim reduction
    for dim_reduction_type in strided_convolution dilated_convolution max_pooling avg_pooling
    do
        # test filters 4 16 32 64
        for num_filters in 4 16 32 64
        do
            mkdir -p "stats/num_filters/"$dim_reduction_type"_"$num_filters"filters"
            python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py \
        	   --batch_size 100 \
        	   --seed 0 \
        	   --num_layers 4 \
        	   --num_filters $num_filters \
        	   --dim_reduction_type $dim_reduction_type \
        	   --experiment_name "stats/num_filters/"$dim_reduction_type"_"$num_filters"filters/"$counter"run" \
        	   --use_gpu True
        done
    done
done
