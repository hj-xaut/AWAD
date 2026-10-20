import argparse
# Training settings
parser = argparse.ArgumentParser(description="Super-Resolution")
parser.add_argument("--upscale_factor", default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=0,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=1, help="training batch size") # 2x 32   4x 16
parser.add_argument("--nEpochs", type=int, default=200, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", default=True, help="show Tensorboard")
parser.add_argument("--log_dir", default='./logs', help="show Tensorboard")
parser.add_argument("--num_bands", default=102, type=int, help="show Tensorboard")
parser.add_argument("--lr", type=int, default=1e-4, help="lerning rate")
parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=0, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="./checkpoint/NET/\harvard/2_256256664/model_4_epoch_.pth", type=str, help="Path to checkpoint (default: none)  checkpoint/model_epoch_95.pth")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument("--datasetName", default="CAVE", type=str, help="data name")
parser.add_argument('--method', default='NET', type=str, help='super resolution method name')

# Network settings
parser.add_argument('--n_module', type=int, default=5, help='number of  modules')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')

# Test image
parser.add_argument('--model_name', default='./model_4_epoch_57.pth', type=str, help='super resolution model name ')

opt = parser.parse_args()
