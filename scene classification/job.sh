python main.py --train 'train' --dataset 'places' \
--task finetune --num_epoches 500 --learning_rate 0.001 &

python main.py --train 'val' --dataset 'places' \
--task validation:./saved_model-100.ckpt

python main.py --train 'train' --dataset 'places' --task finetune --num_epoches 30 --learning_rate 0.001 --batch_size 500


# Try 1
python main.py --train 'train' --dataset 'places' --task train \
--num_epoches 100 --learning_rate 0.001 --batch_size 100

# Try 2
python main.py --train 'train' --dataset 'places' \
--task finetune --num_epoches 500 --learning_rate 0.0001 &


python main.py --train 'train' --dataset 'places' \
--task 'continue_training' --num_epoches 500 --learning_rate 0.00001 &

python main.py --train 'train' --dataset 'places' \
--task finetune --num_epoches 500 --learning_rate 0.0001 &

# Mean, normalization.

# Data augmentation.

scp -i ~/Jianbochen.pem ubuntu@ec2-52-36-247-229.us-west-2.compute.amazonaws.com:~/CS280/HW2/ps3/data/* ./data


scp -i ~/Jianbochen.pem ubuntu@ec2-52-36-247-229.us-west-2.compute.amazonaws.com:~/CS280/HW2/ps3/model4/saved_model-20* .


python main.py --train 'train' --dataset 'places' \
--task 'continue_training' --num_epoches 500 --learning_rate 0.00001 &

python main.py --train 'eval' --dataset 'places' \
--task 'eval' &