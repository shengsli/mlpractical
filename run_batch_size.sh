# run 5 times
for counter in {1..5}
do
    # test 4 dim reduction
    for dim_reduction_type in strided_convolution dilated_convolution max_pooling avg_pooling
    do
        # test batch_size 20 40 100 200
        for batch_size in 20 40 100 200
        do
            mkdir -p "stats/batch_size/"$dim_reduction_type"_"$batch_size"batch_size"
            python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py \
        	   --batch_size $batch_size \
        	   --seed 0 \
        	   --num_layers 4 \
        	   --num_filters 64 \
        	   --dim_reduction_type $dim_reduction_type \
        	   --experiment_name "stats/batch_size/"$dim_reduction_type"_"$batch_size"batch_size/"$counter"run" \
        	   --use_gpu True
        done
    done
done
