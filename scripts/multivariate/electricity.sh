if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_electricity" ]; then
    mkdir ./logs/LongForecasting_electricity
fi

model_name=Conv
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
seq_len=1600
pred_len=96
kernel_size=78
batch_size=16
learning_rate=0.005

python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --channel 321 \
    --kernel_size $kernel_size\
    --train_epochs 100\
    --itr 1\
    --batch_size $batch_size\
    --learning_rate $learning_rate\
    --rev \
    --gpu 0 >logs/LongForecasting_electricity/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$kernel_size'_'$batch_size'_'$learning_rate.log 
