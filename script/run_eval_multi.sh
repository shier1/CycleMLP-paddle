CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg="./configs/cyclemlp_b1.yaml" \
-dataset="imagenet2012" \
-batch_size=8 \
-data_path="./dataset/imagenet" \
-eval \
-pretrained="output/best_cyclemlp301-350/Best_CycleMLP"
