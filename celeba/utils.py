import csv
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from metric import get_all_metrics

def write_all_csv(results, iter_name, column_name, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(([iter_name, column_name]))
        writer.writerows(results)


def write_csv(lists, iter_name, colmun_name, file_name):
    write_all_csv([(i, item) for i, item in enumerate(lists)], iter_name, colmun_name, file_name)


def write_csv_rows(file_name, column_list):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(column_list)


def get_one_hot(y, num_class, device):
    if len(y.shape) == 1:
        y_new = y.unsqueeze(-1)
    else:
        y_new = y
    y_one_hot = torch.FloatTensor(y_new.shape[0], num_class).to(device)
    y_one_hot.zero_()
    y_one_hot.scatter_(1, y_new, 1)
    return y_one_hot


def evaluation(test_loader, predictor, epoch, args, device, square_x=None, square_y=None):
    predictor.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    fxs = []
    fxs_prob = []
    y_all = []
    d_all = []
    test_total_num = 0
    test_true_num = 0
    
    for x, (y_d_l), _ in pbar:
        # Control if you are receiving more than one label
        if len(y_d_l)>1:
            x, y, d = x, y_d_l[0], y_d_l[1]
        else:
            x, y, d = x, y_d_l[0], None
            
        d = d.to(device)
        
        # Transform to a constant variables
        if args.mask_sensitive_attribute:
            # tensor of zeros of size bs, attr_class
            # assume attr_class = 2
            d_one_hot = torch.zeros((d.size(0), 2)).to(device)
        else:
            d_one_hot = get_one_hot(d, 2, device)  # one-hot [bs, attr_class]
            # Scale the sensitive attribute
            d_one_hot = (d_one_hot-torch.tensor([1-test_loader.dataset.z_statistics[0],test_loader.dataset.z_statistics[0]]).to(device)) \
                        /torch.tensor([test_loader.dataset.z_statistics[1], test_loader.dataset.z_statistics[1]]).to(device)
        
        #create mask for images
        if args.mask_square:
            # ipdb.set_trace()
            mask = torch.ones(3, 224, 224, dtype=torch.bool) if args.mask_square_in else torch.zeros(3, 224, 224, dtype=torch.bool)
            mask[:, square_x:square_x+args.square_size, square_y:square_y+args.square_size] = False if args.mask_square_in else True
            
            # Apply mask
            x[:, mask] = 1.0
        
        d_all.append(d)
        y_all.append(y)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            lgt = predictor(x, d_one_hot.half())
            fxs_prob.append(lgt)
        test_total_num += y.shape[0]
        pred = lgt.argmax(1)  # [bs]
        fxs.append(pred)
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Epoch {epoch} Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Epoch {epoch} Acc {100 * test_true_num / test_total_num:.2f}%")
    y_all, d_all = torch.cat(y_all).view(-1).cpu().numpy(), torch.cat(d_all).view(-1).cpu().numpy()
    ds_dict = {"Male": d_all, "Female": 1 - d_all}
    fxs = torch.cat(fxs).view(-1).detach().cpu().numpy()
    fxs_prob = torch.cat(fxs_prob, dim=0).detach().cpu().numpy()
    ret_no_class_balance = get_all_metrics(y_true=y_all, y_pred=fxs, y_prob=fxs_prob, z=ds_dict,use_class_balance=False)
    return ret_no_class_balance, test_true_num / test_total_num


def load_result(result_list, result_loader, accuracy):
    metric_Min_ACC_AcrossZY_no_balance = result_list["Min_ACC_AcrossZY"]
    result_loader[0].append(metric_Min_ACC_AcrossZY_no_balance)
    metric_ED_FPR_AcrossZ_no_balance = result_list["ED_FPR_AcrossZ"]
    result_loader[1].append(metric_ED_FPR_AcrossZ_no_balance)
    metric_ED_FNR_AcrossZ_no_balance = result_list["ED_FNR_AcrossZ"]
    result_loader[2].append(metric_ED_FNR_AcrossZ_no_balance)
    metric_PO_no_balance = result_list["ED_PO1_AcrossZ"]
    result_loader[3].append(metric_PO_no_balance)
    result_loader[4].append(accuracy)
    Loss = result_list['Loss']
    result_loader[5].append(Loss)


def display_result(accuracy, ret_no_class_balance):
    print(f"test acc: {accuracy:.4f}")
    print(f"test AUC: {ret_no_class_balance['AUC']:.4f}")
    print(f"Min_ACC_AcrossZY: {ret_no_class_balance['Min_ACC_AcrossZY'] * 100: .2f}")
    print(f"ED_FPR_AcrossZ: {ret_no_class_balance['ED_FPR_AcrossZ']: .4f}")
    print(f"ED_FNR_AcrossZ: {ret_no_class_balance['ED_FNR_AcrossZ']: .4f}")
    print(
        f"ED_FR_AcrossZ: {(ret_no_class_balance['ED_FPR_AcrossZ'] + ret_no_class_balance['ED_FNR_AcrossZ']): .4f}")
    print(f"ED_PO1_AcrossZ: {ret_no_class_balance['ED_PO1_AcrossZ']: .4f}")

def get_transform(method, image_size, reprogram_size=None, ):
    if method in ["std", "adv", "rpatch", "roptim"]:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])
    elif method == "repro":
        assert reprogram_size is not None
        l_pad = int((image_size - reprogram_size + 1) / 2)
        r_pad = int((image_size - reprogram_size) / 2)

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.Resize(reprogram_size),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
            transforms.RandomHorizontalFlip(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.Resize(reprogram_size),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
        ])
    else:
        raise ValueError

    return transform_train, transform_test


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    torch.backends.cudnn.enabled = False

    # If you set the cudnn.benchmark the CuDNN library will benchmark several algorithms and pick that which it found to be fastest.
    # Rule of thumb: useful if you have fixed input sizes
    torch.backends.cudnn.benchmark = False

    # Some of the listed operations don't have a deterministic implementation. So if torch.use_deterministic_algorithms(True) is set, they will throw an error.
    torch.backends.cudnn.deterministic = True