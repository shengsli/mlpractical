# run 5 times
for counter in {1..5}
do
    # test 4 dim reduction
    for dim_reduction_type in strided_convolution dilated_convolution max_pooling avg_pooling
    do
        # test layer 1..6
        for num_layers in {1..6}
        do
            mkdir -p "stats/num_layers/"$dim_reduction_type"_"$num_layers"layers"
            python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py \
        	   --batch_size 100 \
        	   --seed 0 \
        	   --num_layers $num_layers \
        	   --num_filters 64 \
        	   --dim_reduction_type $dim_reduction_type \
        	   --experiment_name "stats/num_layers/"$dim_reduction_type"_"$num_layers"layers/"$counter"run" \
        	   --use_gpu True
        done
    done
done
