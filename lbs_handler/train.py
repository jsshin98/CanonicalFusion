import torch    
from torch.utils.data import DataLoader
from dataset import LBSUnwrappingDataset
from model import LBSModel
import torch.nn as nn
import json
from tqdm import tqdm
import os 
device = torch.device('cuda')

class LBSLoss(nn.Module):
    def __init__(self, device, loss_type='l1', use_kl=False, use_softmax=False):
        super(LBSLoss, self).__init__()
        self.l1_loss = nn.L1Loss().to(device)
        self.huber_loss = nn.HuberLoss().to(device)

        self.criteria = self.l1_loss if loss_type == 'l1' else self.huber_loss
        
        self.use_kl = use_kl
        self.use_softmax = use_softmax
        
        self.loss_dict = {'loss':0, 'l1':0, 'reg':0, 'sparse':0, 'nonzero':0,}
                          
    def narrow_gaussian(self, x, ell):
        return torch.exp(-0.5 * (x / ell) ** 2)

    def approx_count_nonzero(self, x, ell=1e-1):
        # Approximation of || x ||_0
        return x.shape[1] - self.narrow_gaussian(x, ell).sum(dim=1)
    
    def kl_divergence(self, pred, gt):
        return torch.nn.functional.kl_div(pred, gt, reduction='batchmean')
    
    def eikonal_loss(self, pred, gt):
        return torch.nn.functional.mse_loss(pred, gt)

    def forward(self, pred, gt):
        l1 = self.criteria(pred, gt)
        
        
        sparse = (pred.abs()+1e-12).pow(0.8).sum(1).mean()

        reg = self.l1_loss(torch.ones(gt.shape[0]).to(device), torch.sum(pred, dim=1))        
        nonzero = self.l1_loss(self.approx_count_nonzero(pred), self.approx_count_nonzero(gt))

        loss = l1 * 10 + nonzero * 1 
        
        
        self.loss_dict['loss'] = loss
        self.loss_dict['l1'] = l1
        
        self.loss_dict['nonzero'] = nonzero
    

        if not self.use_softmax:
            self.loss_dict['reg'] = reg
            self.loss_dict['loss'] += reg        
        
        if self.use_kl:
            kl = self.kl_divergence((pred+1e-7).log(), gt)
            self.loss_dict['kl'] = kl
            self.loss_dict['loss'] += kl 
        else:
            self.loss_dict['sparse'] = sparse
            self.loss_dict['loss'] += sparse
                
        return self.loss_dict

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LBS unwrapping')

    
    parser.add_argument('--data-path', type=str, default='data/smplx_1024_UV_LBS.pickle')
    parser.add_argument('--save_dir', type=str, default='./result_uv_1024')
    parser.add_argument('--loss_type', type=str, default='l1')
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--use_softmax', action='store_true')
    parser.add_argument('--use_kl', action='store_true')
    parser.add_argument('--step_lr', type=int, default=0, )
    

    args = parser.parse_args()
        
    configs = {
        'use_softmax': args.use_softmax,
        'use_kl': args.use_kl,
        'lr': args.lr,
        'epoch': args.epoch,
        'batch_size': args.batch_size,
        'loss_type' : args.loss_type,
        'step_lr': args.step_lr,
        'activation': args.activation,
    }

    dataset = LBSUnwrappingDataset(args.data_path)
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

    os.makedirs(args.save_dir, exist_ok=True)

    json.dump(configs, open(os.path.join(args.save_dir, 'config.json'), 'w'))
    objective = LBSLoss(device, loss_type='l1', use_kl=True, use_softmax=True)
    
    model = LBSModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) 
    if args.step_lr > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.7)
    # for i, data in enumerate(dataloader):
    #     x, y = data
        
    #     output = model(x)
        
    best_loss = 100000
    
    for epoch in range(args.epoch):
        for i, (input_lbs, gt_lbs) in enumerate(tqdm(train_loader)):
            input_lbs, gt_lbs = input_lbs.to(device), gt_lbs.to(device)

            pred_lbs = model(input_lbs)

            loss_dict = objective(pred_lbs, gt_lbs)    
            loss = loss_dict['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.print_freq!=-1 and i % args.print_freq == 0:
                loss_str = ["Train:"] + ["{} Loss {:4f} |".format(k, v) for k, v in loss_dict.items() if k != 'loss'] + ["Total Loss {:4f}".format(loss)]
                print(" ".join(loss_str))

                savefilename = os.path.join(args.save_dir) + '/' + str(i) + '.tar'
                
                torch.save({
                    'loss' : loss_dict,
                    'epoch': i,
                    'state_dict': model.state_dict()
                }, savefilename)        

        if i % args.val_epoch == 0 or epoch == args.epoch - 1:
            total_val_loss = {'l1': 0, 'reg': 0, 'sparse': 0, 'nonzero': 0, 'total': 0, 'kl': 0}
                
            with torch.no_grad():
                model.eval()
                for j, (input_lbs, gt_lbs) in enumerate(tqdm(val_loader)):
                    input_lbs, gt_lbs = input_lbs.to(device), gt_lbs.to(device)
                        
                    pred_lbs = model(input_lbs)
                    loss_dict = objective(pred_lbs, gt_lbs)
                        
                    try:
                        for k, v in loss_dict.items():
                            total_val_loss[k] += v
                    except:
                        total_val_loss = loss_dict.copy()
                    
                    
            val_loss_str = [f"Val {epoch} :"] + ["{} Loss {:4f} |".format(k, v/len(val_loader)) for k, v in total_val_loss.items() if k != 'loss'] + ["Total Loss {:4f}".format(total_val_loss['loss']/len(val_loader))]
            print(" ".join(val_loss_str))
                
            if total_val_loss['loss']/len(val_loader) < best_loss:
                best_loss = total_val_loss['loss']/len(val_loader)
                savefilename = os.path.join(args.save_dir) + '/' + 'best.tar'
                loss_forsave = {k: v/len(val_loader) for k, v in total_val_loss.items()}
                
                torch.save({
                    'loss': loss_forsave, 
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                }, savefilename)
            
        model.train()
        if args.step_lr:
            scheduler.step()
            
if __name__ == '__main__':
    main()