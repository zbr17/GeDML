echo "Optional setting:"
echo "1) margin loss"
echo "2) proxy anchor loss"
read -p "Please choose the setting (input the number): " setting

echo "Optional dataset:"
echo "1) cub200"
echo "2) cars196"
echo "3) online_products"
read -p "Please choose the dataset (input the number): " dataset

case $setting in
1)
    # margin loss
    case $dataset in
    1)
        # cub200
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting margin --embeddings_dim 512 \
        --margin_alpha 1 --margin_beta 0.5 --num_classes 100 \
        --lr_trunk 0.00003 --lr_embedder 0.0003 --lr_loss 0.01 \
        --dataset cub200 --delete_old
        ;;
    2)
        # cars196 # TODO:
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting margin --embeddings_dim 512 \
        --margin_alpha 1 --margin_beta 0.5 --num_classes 98 \
        --lr_trunk 0.00003 --lr_embedder 0.0003 --lr_loss 0.01 \
        --dataset cars196 --delete_old
        ;;
    3)
        # online_products # TODO:
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting margin --embeddings_dim 512 \
        --margin_alpha 1 --margin_beta 0.5 --num_classes 11318 \
        --lr_trunk 0.00003 --lr_embedder 0.0003 --lr_loss 0.01 \
        --dataset online_products --delete_old
        ;;
    *)
        echo No matched dataset!
    esac
    ;;
2)
    # proxy anchor loss
    case $dataset in 
    1)
        # cub200
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting proxy_anchor --embeddings_dim 512 \
        --proxyanchor_margin 0.1 --proxyanchor_alpha 32 --num_classes 100 \
        --wd 0.0001 --gamma 0.5 --step 10 \
        --lr_trunk 0.0001 --lr_embedder 0.0001 --lr_collector 0.01 \
        --dataset cub200 --delete_old \
        --warm_up 1 --warm_up_list embedder collector
        ;;
    2)
        # cars196
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting proxy_anchor --embeddings_dim 512 \
        --proxyanchor_margin 0.1 --proxyanchor_alpha 32 --num_classes 98 \
        --wd 0.0001 --gamma 0.5 --step 20 \
        --lr_trunk 0.0001 --lr_embedder 0.0001 --lr_collector 0.01 \
        --dataset cars196 --delete_old \
        --warm_up 1 --warm_up_list embedder collector
        ;;
    3)
        # online_products # TODO:
        CUDA_VISIBLE_DEVICES=0 python demo.py \
        --data_path $WORKSPACE/datasets \
        --save_path $WORKSPACE/exp/GeDML \
        --device 0 --batch_size 180 --test_batch_size 180 \
        --setting proxy_anchor --embeddings_dim 512 \
        --proxyanchor_margin 0.1 --proxyanchor_alpha 32 --num_classes 11318 \
        --wd 0.0001 --gamma 0.5 --step 20 \
        --lr_trunk 0.0001 --lr_embedder 0.0001 --lr_collector 0.01 \
        --dataset online_products --delete_old \
        --warm_up 1 --warm_up_list embedder collector
        ;;
    *)
        echo No matched dataset!
    esac
    ;;
*)
    echo No matched setting!
esac