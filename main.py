import argparse

import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datasetfactory as dfs
import config
import model_finetune
from dataset_batch import BatchData
import os


def parse_option():
    parser = argparse.ArgumentParser('arguements for dataset and training')

    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='mini bath-size')
    parser.add_argument('--num-workers', type=int, default=0, help='numbers parallel workers to use')
    parser.add_argument('--check-point', type=int, default=50, help='check point')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', help='to save checkpoint')
    parser.add_argument('--resume', action='store_true', help='use class moco')
    args = parser.parse_args()
    return args


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Class labels of DCASE SED 2017 task - Events from street scene
__class_labels_dict = {
    'beach': 0,
    'bus': 1,
    'cafe/restaurant': 2,
    'car': 3,
    'city_center': 4,
    'forest_path': 5,
    'grocery_store': 6,
    'home': 7,
    'library': 8,
    'metro_station': 9,
    'office': 10,
    'park': 11,
    'residential_area': 12,
    'train': 13,
    'tram': 14
}

# Development and evaluation sets paths
development_folder = 'data/TUT-acoustic-scenes-2017-development/'
evaluation_folder = 'data/TUT-acoustic-scenes-2017-evaluation/'


def train(model, train_loader, epoch, check_point):
    step = 0
    model.to(device)
    criteria = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)

    for epoch_idx in range(1, epoch + 1):
        model.train()
        sum_loss = 0
        for batch_idx, (mel, target) in enumerate(train_loader):
            optimizer.zero_grad()
            mel, target = mel.to(device), target.to(device)
            logits = model(mel)
            loss = criteria(logits, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1

            if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}'.
                      format(epoch_idx, batch_idx + 1, step, sum_loss / (batch_idx + 1)))

        scheduler.step()


def evaluate(model, test_loader):
    model.to(device)
    model.eval()

    preds_list = []
    target_list = []
    total = 0.0
    correct = 0.0

    for batch_idx, (mel, target) in enumerate(test_loader):
        mel, target = mel.to(device), target.to(device).float()
        out = torch.softmax(model(mel), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        labels = [y.item() for y in target]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

        # Test Accuracy
    test_acc = 100.0 * correct / total
    print('correct: ', correct, 'total: ', total)
    print('Test Accuracy : %.2f' % test_acc)

    return test_acc


if __name__ == '__main__':
    args = parse_option()

    np.random.seed(1900)
    model = model_finetune.Transfer_Cnn14(sample_rate=config.sr, window_size=config.win_len, hop_size=config.hop_len,
                                          mel_bins=config.nb_mel_bands,
                                          fmin=50, fmax=14000, classes_num=15, freeze_base=False).to(device)
    if args.pretrain:
        model.load_from_pretrain(pretrained_checkpoint_path='Cnn14_mAP=0.431.pth')

    development_data = dfs.MelData(development_folder, __class_labels_dict, sample_rate=config.sr,
                                   n_mels=config.nb_mel_bands,
                                   n_fft=config.nfft, hop_length=config.hop_len)

    evaluation_data = dfs.MelData(evaluation_folder, __class_labels_dict, sample_rate=config.sr,
                                  n_mels=config.nb_mel_bands,
                                  n_fft=config.nfft, hop_length=config.hop_len)

    X_dev, Y_dev = development_data.mel_tensor, development_data.label_tensor
    X_eval, Y_eval = evaluation_data.mel_tensor, evaluation_data.label_tensor

    print(X_eval.shape, Y_eval.shape)

    train_loader = torch.utils.data.DataLoader(BatchData(X_dev, Y_dev), batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(BatchData(X_eval, Y_eval), batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)
    if args.resume:
        resume_path = 'cnn14_scratch/checkpoint_base.pth'
        model.load_state_dict(torch.load(resume_path))
        model.train()
    else:
        train(model, train_loader, epoch=args.epoch, check_point=args.check_point)
    if args.save:
        save_path = 'cnn14_scratch'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(),
                   os.path.join(save_path, 'checkpoint_base' + '.pth'))
    evaluate(model, test_loader)

