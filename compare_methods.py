"""
This script is for time series classification task.
"""
import copy
import argparse
from tqdm import tqdm
from joblib import dump, load
# from plot import plot_attention_maps,attention_plot
import torch.optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
import rasterio
from rasterio.transform import from_origin
import os
DATAPATH = Path(r"/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/npy/") 
SEEDS = [1]  # 5 repeated trails
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal prediction with different models")
    parser.add_argument('--npy_path', type=str, default="/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/npy/xiangcheng_test_patch5.npy", help='Directory containing temporal TIFF files')
    parser.add_argument('--model', type=str, default="LTAE",help='select model architecture.')
    parser.add_argument('--model_path', type=str, default = "/nfs/project/netdisk/192.168.100.192/d/private/gaohm/SITS_MoCo/results/LTAE_R2_Seed1_512", help='Path to model checkpoint')
    parser.add_argument('--model_type', choices=['pixel', 'patch'], default = 'pixel', help='Model input type')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for patch-based model')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batchsize', type=int, default= 512, help='Prediction batch size')
    return parser.parse_args()


def acc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datapath = DATAPATH
    seed = SEEDS[0]
    model_path = args.model_path
    loaded_array = np.load(args.npy_path, allow_pickle=True)
    data = loaded_array[0]
    labels = loaded_array[1].astype(int)
    # import ipdb;ipdb.set_trace()
    ndims= 10
    dataset = TABACCO_Crops(data=data,label=labels,nclasses=args.num_classes,datapath=datapath, seed=seed)



   
    print("=> creating model '{}'".format(args.model))
    if args.model in ['rf', 'RF']:
        rf_data = []
        rf_label = dataset.label.tolist()
        for i, X in enumerate(dataset.data):
            X = np.array(X)
            temp = TABACCO_Crops.transform(dataset, X)[0].numpy().reshape(-1)
            rf_data.append(temp)
        # import ipdb;ipdb.set_trace()
        best_model_path = os.path.join(model_path , 'model_best.joblib')
        model = load(best_model_path)

        y_pred = model.predict(rf_data)
        scores = accuracy(rf_label, y_pred, args.num_classes)
        scores_msg = ", ".join(
            [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results : \n\n {scores_msg} \n\n")

    else:
        model = get_model(args.model, ndims, args.num_classes, device)
        best_model_path = os.path.join(model_path, 'model_best.pth')
        print('Restoring best model weights for testing...')
        checkpoint = torch.load(best_model_path)
        state_dict = {k: v for k, v in checkpoint['model_state'].items()}
        criterion = checkpoint['criterion']
        model.load_state_dict(state_dict)
        model.eval()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

        # with torch.no_grad():
        #     y_true_list = list()
        #     y_pred_list = list()
        #     with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        #         for idx, (X, y) in iterator:
        #             # 数据解包 
        #             # import ipdb;ipdb.set_trace()
        #             main_input = X[0].to(device)
        #             doy = X[1].to(device)
        #             ndvi_gt = X[2].to(device)
        #             tsp_gt = X[3].to(device)
        #             y = y.to(device)

        #             logits, ndvi_pred, tsp_pred = model(main_input, doy)  
        #             out = F.log_softmax(logits, dim=-1)
        #             y_true_list.append(y)
        #             y_pred_list.append(out.argmax(-1))
        # y_true = torch.cat(y_true_list).cpu().numpy()
        # y_pred = torch.cat(y_pred_list).cpu().numpy()

        # scores = accuracy(y_true, y_pred, args.num_classes)
        
        with torch.no_grad():
            y_pred = []
            y_true_list = []
            attn_list = []
            with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
                for idx, (X, y) in iterator:
                    X = recursive_todevice(X, device)
                    logits = model(X)
                    # import ipdb;ipdb.set_trace()
                    out = F.log_softmax(logits, dim=-1)
                    y_true_list.append(y)
                    y_pred.append(out.argmax(-1).cpu().numpy())
                    # attn_list.append(attn)

            y_pred = np.concatenate(y_pred, axis=0)
            y_true = torch.cat(y_true_list).cpu().numpy()
            scores = accuracy(y_true, y_pred, args.num_classes)
            # attn = torch.cat(attn_list, dim=1).permute(1, 0, 2).cpu().numpy()

      
                         







def main():
    args = parse_args()
    seeds = SEEDS
    print('seed in', seeds)
    for seed in seeds:
        args.seed = seed
        print(f'Seed = {args.seed} --------------- ')

        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

        acc(args)

if __name__ == '__main__':
    main()
