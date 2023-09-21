import argparse
import os
import time
import numpy
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataset import CelebAFast as CelebA
from models import resnet18, ResidualBlock, AlexNet, OneMLP, TwoMLP, Linear
from utils import *
import warnings

# import ipdb

warnings.filterwarnings("ignore")

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

features_list = ('lefteye,righteye,nose,leftmouth,rightmouth').split(',')

# Use this instead if you want to reduce the name of the files
attr_dict = {}
for i, attr in enumerate(attr_list):
    attr_dict[attr] = i

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/celeba.hdf5')
    parser.add_argument('--result-dir', type=str, default='results/')
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--sensitive-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "alexnet","1mlp","2mlp","linear"])
    parser.add_argument('--step', type=int, default=20, help='step for placing the white square')
    
    #Add arguments for masking sensitive attribute
    parser.add_argument('--mask-sensitive-attribute', action='store_true', default=False)
    
    #Add arguments for masking pixels within a square-mask
    parser.add_argument('--mask-square', action="store_true", default=False)
    parser.add_argument('--mask-square-in', action="store_true", default=False, help='keep (true) or remove (false) the pixels within square-mask')
    parser.add_argument('--square-size', type=int, default=10)
    
    # Use the following to move square-mask. Set x-start=x-end if you don't need to move the square
    parser.add_argument('--square-x-start', type=int, default=0)
    parser.add_argument('--square-x-end', type=int, default=224)
    parser.add_argument('--square-y-start', type=int, default=0)
    parser.add_argument('--square-y-end', type=int, default=224)
    
    # Add arguments for masking specific features from CelebA dataset
    parser.add_argument('--mask-feature', action='store_true', default=False)
    parser.add_argument('--mask-feature-in', action='store_true', default=False, help='keep (true) or remove (false) the pixels within square-mask')
    parser.add_argument('--features', type=str, default='lefteye') # if you want to provide more than 1, separate with comma symbol (,)
    parser.add_argument('--size-mask', type=int, default=20)

    args = parser.parse_args()

    args.sensitive_attrs = args.sensitive_attrs.split(',')
    args.target_attrs = args.target_attrs.split(',')
    args.features = args.features.split(',')
    
    #Check that target, se
    for target_attr in args.target_attrs:
        assert target_attr in attr_list
    for sens_attr in args.sensitive_attrs:
        assert sens_attr in attr_list
    for feature in args.features:
        assert feature in features_list

    return args

def main(args, square_x=None, square_y=None):
    if args.mask_square:
        assert args.mask_square and (square_x is not None and square_y is not None), 'square is true and square_x or square_y are None'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_attr_name = 'std' + "_" + args.model + "_" + "_target"
    attr_targ = ''
    for attr in args.target_attrs:
        # model_attr_name += str(attr_dict[attr]) # use this if you want to use numbers
        model_attr_name += attr
        model_attr_name += "_"
        attr_targ += attr
    
    model_attr_name += "sensitive"
    attr_sens = ''
    for attr in args.sensitive_attrs:
        # model_attr_name += str(attr_dict[attr]) # use this if you want to use numbers
        model_attr_name += attr
        model_attr_name += "_"
        attr_sens += attr
    
    if args.mask_sensitive_attribute:
        model_attr_name += f"mask{attr_sens}"
        model_attr_name += "_"
    if args.mask_square:
        model_attr_name += f"squareatx{str(square_x)}y{square_y}_size{args.square_size}_masksquarekeep{args.mask_square_in}"
        model_attr_name += "_"
    
    if args.mask_feature:
        model_attr_name += "features"
        for feature in args.features:
            model_attr_name += f"{feature}"
        model_attr_name += f"_size{args.size_mask}_maskfeatkeep{args.mask_feature_in}"
        model_attr_name += "_"

    if args.exp_name is not None:
        model_attr_name += f'_{args.exp_name}'

    #Load before adding more to the name of the model
    image_size = 224
    transform_train, transform_test = get_transform(method='std',
                                                    image_size=image_size,
                                                    reprogram_size=None)
    
    #Load classes and features
    num_class = 2 ** len(args.target_attrs)
    attr_class = 2 ** len(args.sensitive_attrs)
    features = [[f'{f}_x', f'{f}_y'] for f in args.features] if args.mask_feature else None
    if features:
        features = [item 
                    for sublist in features 
                    for item in sublist]
    # init model
    if args.model=='resnet18':
        predictor = resnet18(block=ResidualBlock,
                        num_classes=num_class)
    elif args.model=='alexnet':
        predictor = AlexNet(num_classes=num_class)
    elif args.model=='1mlp':
        predictor = OneMLP(num_classes=num_class)
    elif args.model=='2mlp':
        predictor = TwoMLP(num_classes=num_class)
    elif args.model=='linear':
        predictor = Linear(num_classes=num_class)
    
    # Predictor to cuda 
    predictor = predictor.to(device)
    
    # init optimizer
    p_optim = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)
    p_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(p_optim,
                                                        gamma=0.1,
                                                        milestones=[int(0.8 * args.epochs),
                                                                    int(0.9 * args.epochs)])
    

    # Load datasets
    train_set = CelebA(args.data_dir, args.target_attrs, args.sensitive_attrs, land_marks=features, size_mask=args.size_mask,
                        keep_land_marks=args.mask_feature_in, img_transform=transform_train, type="train")
    train_set.y_statistics = [(numpy.array(train_set.labels)[:,train_set.y_index].reshape(-1)*1).mean(),
                                (numpy.array(train_set.labels)[:,train_set.y_index].reshape(-1)*1).std()]
    train_set.z_statistics = [(numpy.array(train_set.labels)[:,train_set.z_index].reshape(-1)*1).mean(),
                                (numpy.array(train_set.labels)[:,train_set.z_index].reshape(-1)*1).std()]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)
    
    val_set = CelebA(args.data_dir, args.target_attrs, args.sensitive_attrs, land_marks=features, size_mask=args.size_mask,
                        keep_land_marks=args.mask_feature_in, img_transform=transform_test, type="val")
    val_set.y_statistics = train_set.y_statistics
    val_set.z_statistics = train_set.z_statistics
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    no_balanced_result = [["Min_ACC_AcrossZY"], ["ED_FPR_AcrossZ"], ["ED_FNR_AcrossZ"],
                        ["ED_PO1_AcrossZ"],["Accuracy"], ['Loss']]
                        
    scaler = GradScaler()
    
    best_SA = 10000.0
    start_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        # training
        predictor.train()
        end = time.time()
        print(f"======================================= Epoch {epoch} =======================================")
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
        total_num = 0
        true_num = 0
        loss = 0
        
        for x, (y_d_l), _ in pbar:
            # Control if you are receiving more than one label
            if len(y_d_l)>1:
                x, y, d = x.to(device), y_d_l[0].to(device), y_d_l[1].to(device)
            else:
                x, y, d = x.to(device), y_d_l[0].to(device), None
            
            # Transform to a constant variables
            if args.mask_sensitive_attribute:
                # tensor of zeros of size bs, attr_class
                d_one_hot = torch.zeros((d.size(0), attr_class)).to(device)
            else:
                d_one_hot = get_one_hot(d, attr_class, device)  # one-hot [bs, attr_class]
                # Scale the sensitive attribute
                d_one_hot = (d_one_hot-torch.tensor([1-train_set.z_statistics[0],train_set.z_statistics[0]]).to(device)) \
                        /torch.tensor([train_set.z_statistics[1], train_set.z_statistics[1]]).to(device)
            
            #create mask for images
            if args.mask_square:
                # ipdb.set_trace()
                mask = torch.ones(3, 224, 224, dtype=torch.bool) if args.mask_square_in else torch.zeros(3, 224, 224, dtype=torch.bool)
                mask[:, square_x:square_x+args.square_size, square_y:square_y+args.square_size] = False if args.mask_square_in else True
                
                # Apply mask
                x[:, mask] = 1.0

            p_optim.zero_grad()

            with autocast():

                lgt = predictor(x, d_one_hot.half())
                pred_loss = nn.functional.cross_entropy(lgt, y)
            
            scaler.scale(pred_loss).backward()
            scaler.step(p_optim)
            scaler.update()
            
            loss += pred_loss.item()

            # results for this batch
            total_num += y.size(0)
            true_num += (lgt.argmax(1) == y.view(-1)).type(torch.float).sum().detach().cpu().item()
            acc = true_num * 1.0 / total_num
            pbar.set_description(f"Training Epoch {epoch} Acc {100 * acc:.2f}%")
        pbar.set_description(f"Training Epoch {epoch} Acc {100 * true_num / total_num:.2f}%")
        
        # ipdb.set_trace()
        p_lr_scheduler.step()

        # evaluating
        print("================= Evaluating on Validation Set =================")
        res, accuracy = evaluation(val_loader, predictor, epoch, args, device, square_x, square_y)
        
        load_result(res, no_balanced_result, accuracy)
        display_result(accuracy, res)
        write_csv_rows(os.path.join(os.path.join(args.result_dir, "csv"), f'{model_attr_name}_epochs.csv'),
                       no_balanced_result)
        
        metric = res['Loss']
        if metric < best_SA:
            print("+++++++++++ Find New Best Min Loss +++++++++++")
            best_SA = metric
            best_so_far = epoch
        
        cp = {"predictor": predictor.state_dict(), #here, the keys are 'weight' and 'bias', consider to change to 'fc.weight' and 'fc.bias' when loading in pve computation
                "p_optim": p_optim.state_dict(),
                "p_lr_scheduler": p_lr_scheduler.state_dict(),
                "epoch": epoch,
                "acc": accuracy,
                "best_SA": best_SA,
                'best_epoch': best_so_far,
                }
        torch.save(cp,
                    os.path.join(os.path.join(args.result_dir, "checkpoints"), f'{model_attr_name}_epoch{epoch}.pth.tar'))

        print(f"Time Consumption for one epoch is {time.time() - end}s")

if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.mask_square:
        for x in range(args.square_x_start, args.square_x_end+1, args.step):
            if x+args.square_size>224:continue
            for y in range(args.square_y_start, args.square_y_end+1, args.step):
                if y+args.square_size>224:continue
                print(f'Training for mask_square at (x,y)=({x},{y}), size={args.square_size}, ' + 
                        f'mask_sensitive_attribute={args.mask_sensitive_attribute}')
                main(args=args,
                    square_x=x, 
                    square_y=y)
                print('=============END=============')
    elif args.mask_feature:
        print(f'Training for feature={args.features}, ' + 
                f'mask_sensitive_attribute={args.mask_sensitive_attribute}')
        main(args=args)
        print('=============END=============')
    
    else:
        print(f'Training for mask_sensitive_attribute')
        main(args=args)
        print('=============END=============')