#bin/bash

# python3 run.py --computease --mode train --lib OwnBaselineCNN --params hp_default 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014 --params hp_default 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014_AvgPool --params hp_default 
# python3 run.py --computease --mode train --lib ResNet50_SunShangetAl --params hp_default 
# python3 run.py --computease --mode train --lib DenseNet --params hp_default 

# python3 run.py --computease --mode train --lib ResNet50_SunShangetAl --params hp_dropna 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014_AvgPool --params hp_dropna 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014 --params hp_dropna 
# python3 run.py --computease --mode train --lib OwnBaselineCNN --params hp_dropna 
# python3 run.py --computease --mode train --lib DenseNet --params hp_dropna

# python3 run.py --computease --mode train --lib OwnBaselineCNN --params hp_l1_loss 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014 --params hp_l1_loss 
# python3 run.py --computease --mode train --lib CNNModel_LiChan2014_AvgPool --params hp_l1_loss 
# python3 run.py --computease --mode train --lib ResNet50_SunShangetAl --params hp_l1_loss 
# python3 run.py --computease --mode train --lib DenseNet --params hp_l1_loss

# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_default
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_default_OwnBaselineCNN
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_LiChan2014
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_600
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_b256
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_b2048 --computease
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_l1_loss_ctx_60
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_dropna
# python3 run.py --mode train --lib T_Sequence --data T_LModule_Seq --params hp_dropna_all

# python3 run.py --mode train --computease --num_worker 3 --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_default
# python3 run.py --mode train --computease --num_worker 3 --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_l1_loss
# python3 run.py --mode train --computease --num_worker 3 --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_OwnBaselineCNN
# python3 run.py --mode train --computease --num_worker 3 --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_CNNModel_LiChan2014
# python3 run.py --mode train --computease --num_worker 3 --lib Sequence_pretrained --data RSO_LModule_Seq --params hp_CNNModel_LiChan2014_AvgPool