import argparse

parser = argparse.ArgumentParser(description='Image Dehazing')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--resume', action='store_true', help='resume from specific checkpoint')

# Data specifications
parser.add_argument('--trainfolder', type=str, help='train dataset name')
parser.add_argument('--b_min', type=str, help='beta min for each training dataset')
parser.add_argument('--b_max', type=str, help='beta max for each training dataset')
parser.add_argument('--valfolder', type=str, help='valid dataset name')
parser.add_argument('--valset', type=str, help='valid dataset Name')
parser.add_argument('--b_minVal', type=str, help='beta min for each validation dataset')
parser.add_argument('--b_maxVal', type=str, help='beta max for each validation dataset')
parser.add_argument('--A_vary', type=int, help='percentage of beta vary across channel')
parser.add_argument('--A_min', type=str, help='atmospheric light min')
parser.add_argument('--A_max', type=str, help='atmospheric light max')
parser.add_argument('--remove_border', type=int, default=10, help='remove border drom indoor image boundary')
parser.add_argument('--patch_size', type=int, help='patch size')

# Model specifications 
parser.add_argument('--hazemodel', help='haze model name')
parser.add_argument('--hazemodel_pt', type=str, help='pre-trained haze model path')
parser.add_argument('--iter', type=int, help='recurrent time-steps')
parser.add_argument('--transmodel', help='transmission model name')
parser.add_argument('--transmodel_pt', type=str, help='pre-trained transmission model path')
parser.add_argument('--atmmodel', help='atmospheric model name')
parser.add_argument('--atmmodel_pt', type=str, help='pre-trained atmospheric model path')


# Training specifications
parser.add_argument('--exp_name', type=str, help='name of the experiment')
parser.add_argument('--train_t', action='store_true', help='set this option to train transmission map estimation')
parser.add_argument('--train_a', action='store_true', help='set this option to train atmospheric light estimation')
parser.add_argument('--train_h', action='store_true', help='set this option to train haze model')
parser.add_argument('--test_every', type=int, help='do test per every N batches')
parser.add_argument('--epochs', type=int, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, help='input batch size for training')

# Optimization specifications
parser.add_argument('--loss', type=str, help='loss function configuration')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--decay', type=str, help='learning rate decay type')
parser.add_argument('--gamma', type=float, help='learning rate decay factor for step decay')

args = parser.parse_args()