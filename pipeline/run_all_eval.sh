#bin/bash

# Vector based approaches
python3 run.py --computease --mode eval --lib OwnBaselineCNN --params hp_default 
python3 run.py --computease --mode eval --lib OwnBaselineCNN --params hp_orig 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014 --params hp_default 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014_AvgPool --params hp_default 
python3 run.py --computease --mode eval --lib ResNet50_SunShangetAl --params hp_default 
# python3 run.py --computease --mode eval --lib DenseNet --params hp_default 

python3 run.py --computease --mode eval --lib OwnBaselineCNN --params hp_dropna 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014 --params hp_dropna 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014_AvgPool --params hp_dropna 
python3 run.py --computease --mode eval --lib ResNet50_SunShangetAl --params hp_dropna 
# python3 run.py --computease --mode eval --lib DenseNet --params hp_dropna

python3 run.py --computease --mode eval --lib OwnBaselineCNN --params hp_l1_loss 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014 --params hp_l1_loss 
python3 run.py --computease --mode eval --lib CNNModel_LiChan2014_AvgPool --params hp_l1_loss 
python3 run.py --computease --mode eval --lib ResNet50_SunShangetAl --params hp_l1_loss 
# python3 run.py --computease --mode eval --lib DenseNet --params hp_l1_loss

# Sequence based approaches
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_default
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_default_OwnBaselineCNN
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_LiChan2014
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_600
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_b256
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_b2048
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_ctx_60
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_dropna
python3 run.py --computease --num_worker 3 --mode eval --lib T_Sequence --data T_LModule_Seq --params hp_dropna_all

python3 run.py --computease --num_worker 3 --mode eval --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_OwnBaselineCNN
python3 run.py --computease --num_worker 3 --mode eval --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_CNNModel_LiChan2014
python3 run.py --computease --num_worker 3 --mode eval --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_CNNModel_LiChan2014_AvgPool
python3 run.py --computease --num_worker 3 --mode eval --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_default
python3 run.py --computease --num_worker 3 --mode eval --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_l1_loss


# python3 run.py --computease --num_worker 3 --mode eval --lib OwnBaselineCNN --params hp_orig --data RSO_LModule_noval