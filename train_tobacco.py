
"""
This script is for time series classification task.
"""
import torch.nn as nn 
import copy
import argparse
from tqdm import tqdm
from joblib import dump, load

import torch.optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *

DATAPATH = Path(r"/nfs/project/netdisk/192.168.100.192/d/baiyq/20240820_yunnan_kaoyan/sample/npy/")  # todo replace your datapath here
SEEDS = [1]  # 5 repeated trails

def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models.')
    parser.add_argument('--model', type=str, default="tempcnn",
                        help='select model architecture.')
    parser.add_argument('--ndims', type=int, default=10,
                        help='number of input channel dimensions')
    parser.add_argument('-c', '--nclasses', type=int, default=2,
                        help='num of classes (default: 20)')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-p', '--patch_size', type=int, default=3,
                        help='patch_size (number of time series patched )') 
    parser.add_argument('-m', '--mode', type=str, default="center",
                        help=' mode in center or patch')   
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='warmup epochs')
    parser.add_argument('--schedule', default=None, nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('-l', '--logdir', type=str, default="./results",
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint')
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Error parsing arguments: {e}")
        parser.print_help()
        import sys
        sys.exit(2)

    args.datapath = DATAPATH


    return args

def train_epoch(model, optimizer, criteria, dataloader, device, args, epoch, total_epochs):
  
    losses = {'total': AverageMeter('Total', ':.2f')}
    
    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (X, y) in iterator:
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            
            # 前向传播
            if args.mode == 'center':
                X = recursive_todevice(X, device)
                y = y.to(device)
                logits = model(X)
                total_loss = criteria(logits, y)
                total_loss.backward()
                optimizer.step()
                iterator.set_description(f"train loss={total_loss:.2f}")

                losses['total'].update(total_loss.item())
                return losses['total'].avg
            
            elif args.mode == 'patch':
                # 数据解包 
                main_input = X[0].to(device)
                doy = X[1].to(device)
                ndvi_gt = X[2].to(device)
                tsp_gt = X[3][:,0,:,:].to(device)
                y = y.to(device)
                logits = model(main_input, doy,tsp_gt,ndvi_gt)
                total_loss = criteria(logits, y)
                # 反向传播
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step() 
                  
                # 更新监控指标
                losses['total'].update(total_loss.item(), main_input.size(0))

                iterator.set_postfix({
                    'Total': losses['total'].avg
                })

                return losses['total'].avg
            
def test_epoch(model,criteria, dataloader, device, args,site_name=None, export_csv=None):  
    losses = {'total': AverageMeter('Total', ':.2f')}
    
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                if args.mode == 'center':
                    X = recursive_todevice(X, device)
                    y = y.to(device)
                    
                    logits = model(X)
                    out = F.log_softmax(logits, dim=-1)

                    total_loss = criteria(logits, y)
                    losses['total'].update(total_loss.item())
                    iterator.set_postfix({
                    'Total': losses['total'].avg
                    })
                    y_true_list.append(y)
                    y_pred_list.append(out.argmax(-1))
                    test_loss = losses['total'].avg
                
                elif args.mode == 'patch':
                    # 数据解包 
                    main_input = X[0].to(device)
                    doy = X[1].to(device)
                    ndvi_gt = X[2].to(device)
                    tsp_gt = X[3][:,0,:,:].to(device)
                    y = y.to(device)
                    logits = model(main_input, doy,tsp_gt,ndvi_gt) 
                    out = F.log_softmax(logits, dim=-1)
                    y_true_list.append(y)
                    y_pred_list.append(out.argmax(-1))
                    
                    # 计算各损失
                    total_loss = criteria(logits, y)
                    # 更新监控指标
                    losses['total'].update(total_loss.item(), main_input.size(0))
                    

                    test_loss = losses['total'].avg
                
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    if export_csv:
        # import ipdb;ipdb.set_trace()
        scores = accuracy(y_true, y_pred,site_name=site_name,export_csv=export_csv)
    else:
        scores = accuracy(y_true, y_pred)

    return test_loss , scores

def train(args):
    print("=> creating dataloader")
    data_loader = get_tabacco_dataloader(args.model, args.datapath,args.batchsize,args.patch_size,args.mode,args.ndims,args.seed)
    # import ipdb;ipdb.set_trace()
    ndims = args.ndims
    num_classes = args.nclasses

    if args.model in ['rf', 'RF']:
        (X_train, y_train),test_dict = data_loader
    else:
        (traindataloader, valdataloader), testdataloaders = data_loader

    print("=> creating model '{}'".format(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, ndims, num_classes, device,args.patch_size)
    
    if args.model in ['RF', 'rf']:
        logdir = Path(args.logdir) / (f'G_RF_R{args.nclasses}_Seed{args.seed}')
        logdir.mkdir(exist_ok=True, parents=True)
        best_model_path = logdir / 'model_best.joblib'

        print('training Random Forest...')
        model.fit(X_train, y_train)
        print(f"saving model to {str(best_model_path)}\n")
        dump(model, best_model_path)

        print('Restoring best model weights for testing...')
        model = load(best_model_path)
        csv_path = logdir / f"{args.model}_testlog.csv"
        file_exists = Path(csv_path).exists()
        if file_exists:
            os.remove(csv_path)
        for site_name,(X_test,y_test) in test_dict.items():
            y_pred = model.predict(X_test)
            print(f"{site_name} Test results: ")

            scores = accuracy(y_pred, y_test,site_name=site_name,export_csv=csv_path)

        # 添加平均值行
        df = pd.read_csv(csv_path)
        avg_row = df.mean(numeric_only=True)
        avg_row['site_name'] = 'Average'
        df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return logdir
    
    model.modelname = f'{model.modelname}_C{num_classes}_{args.batchsize}_{args.learning_rate}_{args.mode}_0414'
    print(f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}")
    model.apply(weight_init)
    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / 'model_best.pth'
    csv_path = logdir / f"{args.model}_testlog.csv"
    print(f"Logging results to {logdir}")

    #loss function
    criterion = nn.CrossEntropyLoss(reduction="mean")  # ce损失
    # 优化器配置
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,  weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=args.epochs*len(traindataloader))

    log = list()
    tabacco_f1_min = 0
    val_loss_min = np.Inf
    print(f"Training {model.modelname}")
    writer = SummaryWriter(logdir)
    for epoch in range(args.epochs):
        # if args.warmup_epochs > 0:
        #     if epoch == 0:
        #         lr = args.learning_rate * 0.1
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #     elif epoch == args.warmup_epochs:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = args.learning_rate
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device, args, epoch+1,args.epochs)
        val_loss,scores = test_epoch(model,criterion, valdataloader, device, args)
        scores_msg = ", ".join(
            [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)
        scores["epoch"] = epoch + 1
        scores["trainloss"] = train_loss
        scores["testloss"] = val_loss
        log.append(scores)
        tabacco_f1 = scores.pop('class_f1')[1]
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/OA', scores['oa'], epoch)
        writer.add_scalar('Accuracy/F1', tabacco_f1, epoch)
        writer.add_scalar('Accuracy/kappa', scores['kappa'], epoch)


        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(Path(logdir) / "trainlog.csv")

        # if tabacco_f1 > tabacco_f1_min:
        #     tabacco_f1_min = tabacco_f1
        #     save(model, path=best_model_path, criterion=criterion)
        #     print(f'best class f1 in epoch {epoch +1}\n')
        if val_loss < val_loss_min:
            not_improved_count = 0
            save(model, path=best_model_path, criterion=criterion)
            val_loss_min = val_loss
            print(f'lowest val loss in epoch {epoch + 1}\n')
        else:
            not_improved_count += 1
        if not_improved_count >= 40:
            print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
            writer.close()
            break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")
            writer.close()

    # # test
    print('Restoring best model weights for testing...')
    checkpoint = torch.load(best_model_path)
    state_dict = {k: v for k, v in checkpoint['model_state'].items()}
    criterion = checkpoint['criterion']
    torch.save({'model_state': state_dict, 'criterion': criterion}, best_model_path)
    model.load_state_dict(state_dict)
    
    for site_name,testdataloader in testdataloaders.items():
        print(f"{site_name} Test results: ")
        test_loss,scores = test_epoch(model,criterion, testdataloader, device, args,site_name=site_name,export_csv=csv_path)
    # 添加平均值行
    df = pd.read_csv(csv_path)
    avg_row = df.mean(numeric_only=True)
    avg_row['site_name'] = 'Average'
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)
    df.to_csv(csv_path, index=False)    

    return logdir




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

        logdir = train(args)


if __name__ == '__main__':
    main()


