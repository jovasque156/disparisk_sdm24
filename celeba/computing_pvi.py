import argparse
import os
import time
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataset import CelebAFast as CelebA
from models import resnet18, ResidualBlock, AlexNet, OneMLP, TwoMLP, Linear
from utils import *
from fairness import demographic_parity_dif, accuracy
import warnings

import ipdb

warnings.filterwarnings("ignore")

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

features_list = ('all,empty,lefteye,righteye,nose,leftmouth,rightmouth,square').split(',')

attr_dict = {}
for i, attr in enumerate(attr_list):
    attr_dict[attr] = i

insufficient_attr_list = '5_o_Clock_Shadow,Goatee,Mustache,Sideburns,Wearing_Necktie'.split(',')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/celeba.hdf5')
    parser.add_argument('--result-dir', type=str, default='results/')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "alexnet","1mlp","2mlp","linear"])
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--epoch-end', type=int, default=9)
    parser.add_argument('--save-pvi', action='store_true', default=False, help='use to save pvi')
    parser.add_argument('--save-perfo', action='store_true', default=False, help='use to save performances on test dataset')
    
    #Add arguments for masking sensitive attribute and some square of the image
    parser.add_argument('--from-x', type=str, default="all") #choices=['all', 'lefteye','righteye', 'nose', 'leftmouth', 'rightmouth', 'square'])
    
    # This is for the choice square
    parser.add_argument('--square-at-x', type=int, default=0)
    parser.add_argument('--square-at-y', type=int, default=0)
    
    # This is only for choices distinct from all
    parser.add_argument('--mask-in-x', action='store_true', default=False, help='choice of from x is given (true) or note (false)')
    
    parser.add_argument('--to-y', type=str, default="Male", choices=['Male', 'Blond_Hair'])
    parser.add_argument('--size-mask-X', type=int, default=20)
    
    
    
    parser.add_argument('--given-C', type=str, default="empty") # choices=['empty', 'lefteye','righteye', 'nose', 'leftmouth', 'rightmouth', 'square'])
    
    # This is for the choice square
    parser.add_argument('--square-at-x-C', type=int, default=0)
    parser.add_argument('--square-at-y-C', type=int, default=0)
    #This is only for choices distinct of empty
    parser.add_argument('--mask-in-C', action='store_true', default=False, help='choice of from C is given (true) or note (false)')
    
    # for every size-mask
    parser.add_argument('--size-mask-C', type=int, default=20)
    
    args = parser.parse_args()

    args.from_x = args.from_x.split(',')
    for feature in args.from_x:
        assert feature in features_list
    args.given_C = args.given_C.split(',')
    for feature in args.given_C:
        assert feature in features_list
        
        
    return args
    
def compute_pvi(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    image_size = 224
    transform_test = get_transform(method='std',
                                    image_size=image_size,
                                    reprogram_size=None)[1]
    
    #Load classes and features
    num_class = 2 # ** len(args.target_attrs)
    attr_class = 2 # ** len(args.sensitive_attrs)
    if ('all' in args.from_x) or ('square' in args.from_x):
        features_X = None
    else:
        features_X = [[f'{f}_x', f'{f}_y'] for f in args.from_x]    
        features_X = [c 
                    for coordinates in features_X 
                    for c in coordinates]
                    
    
    if ('empty' in args.given_C) or ('square' in args.given_C): 
        features_C = None
    else:
        features_C = [[f'{f}_x', f'{f}_y'] for f in args.given_C] 
        features_C = [c 
                    for coordinates in features_C
                    for c in coordinates]
    
    # Load g
    if 'Male' in args.to_y:
        # target = str(attr_dict['Male'])
        target = 'Male'
        # sensitive = str(attr_dict['Blond_Hair'])
        sensitive = 'Blond_Hair'
        sensitive_attrs = 'Blond_Hair'
    else:
        # target = str(attr_dict['Blond_Hair'])
        target = 'Blond_Hair'
        # sensitive = str(attr_dict['Male'])
        sensitive = 'Male'
        sensitive_attrs = 'Male'
    
    # Load set for H_v_C
    if 'empty' in args.given_C:
        test_set_C = CelebA(args.data_dir, args.to_y, sensitive_attrs, land_marks=None, size_mask=None,
                        keep_land_marks=None, img_transform=transform_test, type="test")
    else:
        test_set_C = CelebA(args.data_dir, args.to_y, sensitive_attrs, land_marks=features_C, size_mask=args.size_mask_C,
                        keep_land_marks=args.mask_in_C, img_transform=transform_test, type="test")
    test_loader_C = DataLoader(test_set_C, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    # Load set for H_v_X
    if 'all' in args.from_x:
        test_set_X = CelebA(args.data_dir, args.to_y, sensitive_attrs, land_marks=None, size_mask=None,
                        keep_land_marks=None, img_transform=transform_test, type="test")
    else:
        test_set_X = CelebA(args.data_dir, args.to_y, sensitive_attrs, land_marks=features_X, size_mask=args.size_mask_X,
                        keep_land_marks=args.mask_in_x, img_transform=transform_test, type="test")
    test_loader_X = DataLoader(test_set_X, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    
    # Init models
    if 'resnet18' in args.model:
        g = resnet18(block=ResidualBlock,
                                num_classes=num_class)
        g_prime = resnet18(block=ResidualBlock,
                                num_classes=num_class)
    elif 'alexnet' in args.model:
        g = AlexNet(num_classes=num_class)
        g_prime = AlexNet(num_classes=num_class)
    elif '1mlp' in args.model:
        g = OneMLP(num_classes=num_class)
        g_prime = OneMLP(num_classes=num_class)
    elif '2mlp' in args.model:
        g = TwoMLP(num_classes=num_class)
        g_prime = TwoMLP(num_classes=num_class)
    elif 'linear' in args.model:
        g = Linear(num_classes=num_class) 
        g_prime = Linear(num_classes=num_class) 
    
    #Load weights
    model_attr_name_base = 'std' + "_" + args.model + "_" + "_target"
    model_attr_name_base += str(target)
    model_attr_name_base += "_"
    model_attr_name_base += "sensitive"
    model_attr_name_base += str(sensitive)
    model_attr_name_base += "_"
    model_attr_name_base += f"mask{str(sensitive)}"
    model_attr_name_base += "_"
    
    model_from_X = model_attr_name_base
    model_from_C = model_attr_name_base
    
    
    if 'all' not in args.from_x:
        if 'square' in args.from_x:
            model_from_X += f"squareatx{args.square_at_x}y{args.square_at_x}_size{args.size_mask_X}_masksquarekeep{args.mask_in_x}"
            model_from_X += "_"
        else:
            model_from_X += "features"
            for feature in args.from_x:
                model_from_X += f"{feature}"
            model_from_X += f"_size{args.size_mask_X}_maskfeatkeep{args.mask_in_x}"
            model_from_X += "_"
        
    
    if 'empty' in args.given_C:
        model_from_C += f"squareatx0y0_size224_masksquarekeepFalse"
        model_from_C += "_"
    elif 'square' in args.given_C:
        model_from_C += f"squareatx{args.square_at_x_C}y{args.square_at_y_C}_size{args.size_mask_C}_masksquarekeep{args.mask_in_C}"
        model_from_C += "_"
    else:
        model_from_C += "features"
        for feature in args.given_C:
            model_from_C += f"{feature}"
        model_from_C += f"_size{args.size_mask_C}_maskfeatkeep{args.mask_in_x}"
        model_from_C += "_"
        
    results = pd.DataFrame()
    for e in range(args.epoch_start, args.epoch_end+1):
        print(f'=====STARTING FOR EPOCH {e}====')
        checkpoint = torch.load(f'/checkpoints/{model_from_X}_epoch{e}.pth.tar', map_location=device)
        g.load_state_dict(checkpoint["predictor"])
        checkpoint = torch.load(f'/checkpoints/{model_from_C}_epoch{e}.pth.tar', map_location=device)
        g_prime.load_state_dict(checkpoint["predictor"])
        
        g.to(device)
        g_prime.to(device)
        
        m = test_set_X.lens
        
        g.eval()
        g_prime.eval()
        
        H_v_C = 0
        pvi_C = []
        for x, (y_d_l), index in test_loader_X:
            x, y, d = x.to(device), y_d_l[0].to(device), y_d_l[1].to(device)
                    
            d_one_hot = torch.zeros((y.size(0), attr_class)).to(device)
            if 'empty' in args.given_C:
                mask = torch.ones(3, 224, 224, dtype=torch.bool)
                # Apply mask
                x[:, mask] = 1.0
            
            if 'square' in args.given_C:
                mask = torch.ones(3, 224, 224, dtype=torch.bool) if args.mask_in_C else torch.zeros(3, 224, 224, dtype=torch.bool)
                mask[:, args.square_at_x_C:args.square_at_x_C+args.size_mask_C, args.square_at_y_C:args.square_at_y_C+args.size_mask_C] = False if args.mask_in_C else True
                
                # Apply mask
                x[:, mask] = 1.0
            
            with torch.no_grad():
                output_g_prime = torch.nn.functional.softmax(g_prime(x, d_one_hot.half()), dim=1)
            
            pred = output_g_prime.argmax(1)
            for i in range(len(y)):
                H_v_C = H_v_C - (1/m)*torch.log(output_g_prime[i,y[i]]).item()
                pvi_index = torch.log(output_g_prime[i,y[i]]).item()
                pvi_C.append((index[i].item(), pvi_index, y[i].item(), pred[i].item(), d[i].item()))
        
        H_v_X = 0
        pvi_X = []
        for x, (y_d_l), index in test_loader_X:
            x, y, d = x.to(device), y_d_l[0].to(device), y_d_l[1].to(device)
                    
            d_one_hot = torch.zeros((y.size(0), attr_class)).to(device)
            
            if 'square' in args.from_x:
                mask = torch.ones(3, 224, 224, dtype=torch.bool) if args.mask_in_x else torch.zeros(3, 224, 224, dtype=torch.bool)
                mask[:, args.square_at_x:args.square_at_x+args.size_mask_X, args.square_at_y:args.square_at_y+args.size_mask_X] = False if args.mask_in_x else True
                
                # Apply mask
                x[:, mask] = 1.0
            
            with torch.no_grad():
                output_g = torch.nn.functional.softmax(g(x, d_one_hot.half()), dim=1)
            
            pred = output_g.argmax(1)
            for i in range(len(y)):
                H_v_X = H_v_X - (1/m)*torch.log(output_g[i,y[i]]).item()
                pvi_index = torch.log(output_g[i,y[i]]).item()
                pvi_X.append((index[i].item(), pvi_index, y[i].item(), pred[i].item(), d[i].item()))
                
        # Compute Demographic Parity
        pred = np.array([i[3] for i in pvi_X])
        real_Y = np.array([i[2] for i in pvi_X])
        S = np.array([i[-1] for i in pvi_X])
        demp = demographic_parity_dif(pred, S, 0)
        acc = accuracy(real_Y, pred)
        
        results = pd.concat([results, 
                            pd.DataFrame({'epoch': [e],
                                            'acc': [acc],
                                            'demp': [demp],
                                            'H_v_X': [H_v_X],
                                            'H_v_C': [H_v_C],
                                            'I_v(X->L|C)': [H_v_C-H_v_X]
                            })],
                            axis=0, ignore_index=True)
        
        print(f'H_v_X= {H_v_X}')
        print(f'H_v_C= {H_v_C}')
        print(f'I_v(X->L|C): {H_v_C-H_v_X}')
        print(f'demp: {demp}')
        print(f'acc: {acc}')
        print()
        
        if args.save_perfo:
            results.to_csv(f'{args.result_dir}performances/{args.model}/perfo_{args.model}_target{target}_from{args.from_x}_given{args.given_C}_epoch{e}.csv')
        
        if args.save_pvi:
            index_X = np.array([i[0] for i in pvi_X])
            pvi_from_X = np.array([i[1] for i in pvi_X])
            
            index_C = np.array([i[0] for i in pvi_C])
            pvi_from_C = np.array([i[1] for i in pvi_C])
            
            pvi_from_C_sorted = np.array([pvi_from_C[np.where(index_C==i)[0][0]] for i in index_X])
            
            prediction = np.array([i[3] for i in pvi_X])
            real_Y = np.array([i[2] for i in pvi_X])
            sensitive = np.array([i[4] for i in pvi_X])
            
            pvi = np.array([-pvi_from_C_sorted[i]+pvi_from_X[i] for i in index_X])
            
            pd.DataFrame({'index': index_X,
                            'real_Y': real_Y,
                            'pred': prediction,
                            'sensitive': sensitive,
                            'piv_from_C': pvi_from_C_sorted,
                            'pvi_from_X': pvi_from_X,
                            'pvi': pvi}).to_csv(f'{args.result_dir}pvis/{args.model}/pvis_{args.model}_target{target}_from{args.from_x}_given{args.given_C}_epoch{e}.csv')
        
        
    print('====END====')
    print(results)

if __name__ == '__main__':
    args = get_args()
    print(args)
    compute_pvi(args)