import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np # Added for MixUp
from tqdm import tqdm
from mnext import MXNet

# --- MixUp Helper Functions ---
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# ------------------------------

def get_cifar_datasets(dataset_name='cifar10'):
    """Get CIFAR-10 or CIFAR-100 datasets with augmentation"""
    
    # Data augmentation for CIFAR
    train_transform_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])
    
    test_transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    # Data augmentation for CIFAR 100
    train_transform_cifar100 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        #AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    test_transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform_cifar10
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform_cifar10
        )
        num_classes = 10
    else:  # cifar100
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform_cifar100
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform_cifar100
        )
        num_classes = 100
    
    return train_dataset, test_dataset, num_classes

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with MixUp"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar for training
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', 
                      leave=False, ncols=100)
    
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # --- MixUp Implementation ---
        # Alpha=0.2 is standard for CIFAR
        data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2, use_cuda=(device.type=='cuda'))
        data, targets_a, targets_b = map(torch.autograd.Variable, (data, targets_a, targets_b))
        
        output = model(data)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        # ----------------------------
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy (Approximation for MixUp)
        _, predicted = output.max(1)
        total += target.size(0)
        # Acc is weighted average of correct predictions for both targets
        correct += (lam * predicted.eq(targets_a).sum().float()
                    + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        
        # Update progress bar
        current_acc = 100. * correct / total
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, test_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar for validation
    val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1} [Val]', 
                    leave=False, ncols=100)
    
    with torch.no_grad():
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            val_pbar.set_postfix({
                'Loss': f'{test_loss/(val_pbar.n+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='MobileNeXt CIFAR Training')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                        help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    # Updated default LR for SGD
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='learning rate (default: 0.1 for SGD)')
    # Updated default weight decay
    parser.add_argument('--weight_decay', type=float, default=4e-5,
                        help='weight decay (default: 4e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='minimum learning rate for cosine annealing (default: 1e-6)')
    parser.add_argument('--width_mult', type=float, default=1.0,
                        help='width multiplier for model (default: 1.0)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', default='./checkpoints', type=str,
                        help='directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get datasets
    train_dataset, test_dataset, num_classes = get_cifar_datasets(args.dataset)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Model
    model = MXNet(num_classes=num_classes, width_mult=args.width_mult)
    model = model.to(device)
    
    # --- Optimize Config (Separate Weight Decay) ---
    # Tắt weight decay cho Alpha, Bias, và BN
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Alpha (scale-aware) không nên bị decay về 0
        if 'alpha' in name or 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    criterion = nn.CrossEntropyLoss()
    
    # Switch to SGD + Momentum (Better generalization for Rep-Models)
    optimizer = optim.SGD(
        param_groups, 
        lr=args.lr, 
        momentum=0.9,
        nesterov=True
    )
    
    # Learning rate scheduler - PyTorch's CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    start_epoch = 0
    best_acc = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    print(f"Training MobileNeXt on {args.dataset.upper()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Optimizer: SGD (lr={args.lr}, momentum=0.9, decay={args.weight_decay})")
    print(f"Tech: Separate Weight Decay + MixUp (alpha=0.2)")
    print("-" * 80)
    
    # Create main progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc='Training Progress', 
                      position=0, leave=True, ncols=120)
    
    # Training loop
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update main progress bar
        epoch_pbar.set_postfix({
            'LR': f'{current_lr:.6f}',
            'Train_Loss': f'{train_loss:.4f}',
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Loss': f'{val_loss:.4f}',
            'Val_Acc': f'{val_acc:.2f}%',
            'Best_Acc': f'{best_acc:.2f}%'
        })
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.save_dir, f'mobilenext_{args.dataset}_latest.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, f'mobilenext_{args.dataset}_best.pth'))
            tqdm.write(f' New best accuracy: {best_acc:.2f}%')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'mobilenext_{args.dataset}_epoch_{epoch+1}.pth'))
            tqdm.write(f' Checkpoint saved at epoch {epoch+1}')
    
    epoch_pbar.close()
    print(f'\n Training completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()