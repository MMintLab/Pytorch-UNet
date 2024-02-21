import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, ToolDataset, full_dataset
from utils.dice_score import dice_loss
import itertools

dir_img = Path('./data_default/imgs/')
dir_mask = Path('./data_default/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        train_set,
        val_set,
        test_set,
        test_unseen_set,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        default: bool = False
):

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size) #, num_workers=os.cpu_count() , pin_memory=True
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    test_unseen_set_loader = DataLoader(test_unseen_set, shuffle=False, drop_last=True, **loader_args)

    n_train = len(train_set)
    n_val = len(val_set)

    # (Initialize logging)
    str_learning_rate = "{:.10f}".format(float(learning_rate)).rstrip('0')
    model_name = 'model_0_' + dataset_name + '_E' + str(epochs) + '_B' + str(batch_size) + '_LR' + str_learning_rate
    print(model_name)
    experiment = wandb.init(project='U-Net', name=model_name, id=model_name, resume='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum) #, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # import pdb; pdb.set_trace()
    # 5. Begin training
    # import pdb; pdb.set_trace()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # batch = next(iter(train_loader))
            # for i in range(2000):
            for batch in train_loader:
                # print(i)
                # import pdb; pdb.set_trace()
                images, true_masks = batch['image'], batch['mask']
                

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # import pdb; pdb.set_trace()

                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                # import pdb; pdb.set_trace()
                masks_pred = model(images)
                # import pdb; pdb.set_trace()
                if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        train_score, train_loss = evaluate(model, train_loader, device, amp)
                        val_score, val_loss = evaluate(model, val_loader, device, amp)
                        test_score, test_loss = evaluate(model, test_loader, device, amp)
                        test_unseen_score, test_unseen_loss = evaluate(model, test_unseen_set_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'train Dice': train_score,
                                'validation Dice': val_score,
                                'test Dice': test_score,
                                'test unseen Dice': test_unseen_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                'train loss_2': train_loss,
                                'validation loss': val_loss,
                                'test loss': test_loss,
                                'test unseen loss': test_unseen_loss,
                                **histograms
                            })
                        except:
                            pass

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     if default:
        #         state_dict['mask_values'] = dataset.mask_values
        #     else:
        #         state_dict['mask_values'] = [0, 255]

        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
    
        models_path = './models/'
        if epoch in [5, 10, 20, 30]:
            model_name = 'model_' + dataset_name + '_E' + str(epoch) + '_B' + str(batch_size) + '_LR' + str_learning_rate
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            torch.save(model.state_dict(), models_path + model_name + '.pth')

    experiment.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--default', action='store_true', default=False, help='Save checkpoints')
    parser.add_argument('--dataset_name', type=str, default='1-tool', help='Name of the dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Name of the device')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1')

    if args.dataset_name == '1-tool':
        device = torch.device('cuda:2')
    elif args.dataset_name == 'train-tools':
        device = torch.device('cuda:0')
    elif args.dataset_name == 'train-test-tools':
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:2')
        
    logging.info(f'Using device {device}')

    if args.device:
        device = torch.device(args.device)
        logging.info(f'Using device {device}')
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    # Change here to adapt to your data
    if args.default:
        n_channels=3
    else:
        n_channels=1
    
    # try:
    # import pdb; pdb.set_trace()
    if args.default:
        print('Using default dataset')
        dataset_name = 'default'
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, args.img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, args.img_scale)

        # # 2. Split into train / validation partitions
        n_val = int(len(dataset) * args.val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
            
    else:
        # 1. Create dataset
        dataset_name = args.dataset_name
        train_set, val_set, test_set, test_unseen_set, _ = full_dataset(dataset_name, device = device)

    # Define your hyperparameters
    hyperparameters = {
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01],
        'batch_size': [args.batch_size],
        'epochs': [30],
    }

    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*hyperparameters.values()):
        # Unpack the hyperparameters
        learning_rate, batch_size, epochs = params

        # Reset the model for each combination of hyperparameters
        model = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)
        model.to(device=device)

        # Train the model with the current hyperparameters
        train_model(
            model=model,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            test_unseen_set=test_unseen_set,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            amp=args.amp,
        )

    # Evaluate the model and save the results
    # ...

    # Update the best hyperparameters based on the evaluation results
    # ...
