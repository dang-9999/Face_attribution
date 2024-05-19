import os
import time
import sys
sys.path.append('.')
import yaml
import torch
import torch.utils.data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from loguru import logger
import torchvision.transforms as transforms
import torch.nn as nn
from utli import *
from loss import Loss
from model import create_net
# from lib.models.query2label import build_q2l
from arg import get_argparse
import Face_dataset as Dataloader
from torchmetrics.functional import accuracy
import csv
import pandas as pd
from torchsummary import summary

# python /home/dang/experiment/Face_Attribute/light_model/main.py --randAug --bn1d --back resnet50 --arch L2L --dataset LFWA --head norm -p 391
# python /home/dang/experiment/Face_Attribute/light_model/main.py --randAug --bn1d --back resnet50 --arch L2L --dataset LFWA --head norm -p 391 --eval-only True
# python /home/dang/experiment/Face_Attribute/light_model/main.py --randAug --bn1d --back resnet50 --arch L2L --dataset LFWA --head norm -p 391 --eval-only --eval-ckpt /home/dang/experiment/model/[resnet50,down]L2L_model_best.pth

def append_to_csv(filename, data):
    # CSV 파일 열기 (추가 모드)
    with open(filename, 'a', newline='') as csvfile:
        # CSV 파일을 쓰기 위한 writer 객체 생성
        writer = csv.writer(csvfile)
        # 데이터 쓰기
        writer.writerow(data)


def main(args):
    # record for the experiment
    global writer
    exp_path = os.path.join('exps/', time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(exp_path, exist_ok=True)
    logger.add(os.path.join(exp_path, 'log.txt'))
    writer = SummaryWriter(logdir=os.path.join(exp_path, 'err'))

    seedForExp(args)
    logger.info('Epochs:', args.epochs, 'seed:', args.seed, 'Bs:', args.batch_size, "beta:", args.beta, "mp:", args.mp)
    logger.info("Use GPU: {} for training".format(args.gpu))
    logger.info(args)
    
    # 모델 선택
    best_err = 100
    if args.arch == 'Q2L':
        # model = build_q2l()
        print(",.")
    elif args.arch == 'L2L':
        model = create_net(args)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)     
        criterion = Loss(mask_ratio=args.ratio).cuda(args.gpu)
        # load_matched_state_dict(model, torch.load('/home/dang/experiment/model/[Mobile,Finetuned]L2L_model_best.pth',map_location='cpu'))
        
    
    # Dataloader
    
    # train_path = '/home/dang/experiment/Face_Attribute/dataset/celeba'
    # val_path = 'tmp'
    # test_path = 'tmp'
    img_path = '/home/dang/experiment/Face_Attribute/dataset'
    img_csv = '/home/dang/experiment/Face_Attribute/dataset/downstream_Train.csv'


    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform= transforms.Compose([
    #             transforms.Resize((112 , 112)),  # 이미지 크기 조정
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 이미지 크기를 모델의 입력 크기에 맞게 조정
    transforms.ToTensor(),           # 이미지를 Torch 텐서로 변환
])

    train_loader = Dataloader.Dataload(csv_file=img_csv, img_path = img_path, transform = transform, batch_size = args.batch_size, state='train')
    # val_loader = Dataloader.Dataload(csv_file=img_csv, img_path = img_path, transform = transform, batch_size = args.batch_size , state='valid')
    test_loader = Dataloader.Dataload(csv_file=img_csv, img_path = img_path, transform = transform, batch_size = args.batch_size , state='test')
    
    # model.qhead.classifier = nn.Linear(512, 3).to('cuda')
    # model.lhead.classifier = nn.Linear(512, 3).to('cuda')
    # print(summary(model, (3, 112, 112)))
    # print(model)
    #====== for downstream task
    # 모델의 파라미터 중 마지막 레이어를 제외한 모든 파라미터의 requires_grad를 False로 설정
    for name, param in model.named_parameters():
        if not name.startswith('qhead.classifier') and not name.startswith('lhead.classifier'):
            param.requires_grad = False
    
    # ======origin version for downstream task
    # for name, param in model.named_parameters():
    #     if 'classifier' not in name:
    #         param.requires_grad = False
            
    print(model)
            


    optimizer = create_optimizer(args, [{'params': model.parameters(), 'lr': args.lr_ft}])
    model.eval()
    if args.eval_only:
        # 컬럼 만들기
        result = pd.DataFrame([], columns=["acc"])
        result.to_csv("/home/dang/experiment/Face_Attribute/result/origin_down_acc.csv", index=False)
        logger.info(f'Load model from {args.eval_ckpt}')
        model.load_state_dict(torch.load(args.eval_ckpt)['state_dict'])
        test(test_loader, model, criterion, 0, args)
        exit()
    
    for epoch in range(args.start_epoch, args.epochs):        
        
        lr = adjust_learning_rate(optimizer, epoch, args, args.lr_ft)
        terr, tloss = train(train_loader, model, criterion, optimizer, epoch, args)
        error, loss = validate(test_loader, model, criterion, 0, args)
        
        # remember best err and save checkpoint
        is_best = error < best_err
        best_err = min(error, best_err)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_err,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args)
    logger.info(best_err)
    with open(os.path.join(exp_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f)
    os.rename(exp_path, exp_path + '_%.2f'%best_err)
    return best_err

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    err = AverageMeter('Error', ':6.2f')
    progress = ProgressMeter(len(train_loader),
        [batch_time, data_time, losses, err], 
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    
    end = time.time()
    beta = torch.distributions.beta.Beta(args.beta, args.beta) if args.beta != 0 else 0
    for i, (images, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True).float()
        target = target.cuda(args.gpu, non_blocking=True).float()
        
        r = torch.rand(1)
        if args.beta > 0 and r < args.mp:
            lam = beta.sample()
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            images = lam*images + (1-lam)*images[rand_index,:]
            if args.arch == 'Q2L':
                pred = model(images)
                loss = args.qratio * (criterion(pred, target_a) * lam + criterion(pred, target_b) * (1 - lam))
            elif args.arch == 'L2L':
                pred_q, pred, mask, r = model(images, target_a, target_b, lam)
                q_target = torch.where(pred_q>0, 1, 0)
                loss = args.qratio * (criterion(pred_q, target_a) * lam + criterion(pred_q, target_b) * (1 - lam))
                if args.head == 's2s':
                    loss += args.lratio * (criterion(pred, q_target, mask) * lam + criterion(pred, q_target[rand_index], mask) * (1 - lam))
                if args.head == 's2l':
                    p = torch.sigmoid(pred)
                    q = torch.sigmoid(pred_q)
                    loss += args.lratio*F.kl_div(torch.log(q),p,reduction='none').mean()
                if args.head == 'r2r':
                    loss += args.lratio*F.mse_loss(r, pred.transpose(0,1))
                    pred = target
                if args.head == 'norm' or args.head == 'enc':
                    loss += args.lratio*(criterion(pred, target_a, mask) * lam + criterion(pred, target_b, mask) * (1-lam))
        else:
            if args.arch == 'Q2L':
                pred = model(images)
                loss = args.qratio*criterion(pred, target)
            elif args.arch == 'L2L':
                pred_q, pred, mask, r = model(x=images)
                loss = args.qratio*criterion(pred_q, target)
                q_target = torch.where(pred_q>0, 1, 0)
                if args.head == 's2s':
                    loss += args.lratio*criterion(pred, q_target, mask)
                if args.head == 's2l':
                    p = torch.sigmoid(pred)
                    q = torch.sigmoid(pred_q)
                    loss += args.lratio*F.kl_div(torch.log(q),p,reduction='none').mean()
                if args.head == 'r2r':
                    loss += args.lratio*F.mse_loss(r, pred.transpose(0,1))
                    pred = target
                if args.head == 'norm' or args.head == 'enc':
                    loss += args.lratio*criterion(pred, target, mask)
        if loss != loss:
            logger.warning('nan loss')
            sys.exit()
        res = torch.where(pred > 0, torch.ones_like(pred), torch.zeros_like(pred))
        
       
        err_batch = 100 - torch.sum(res == target) * (100 / (target.shape[0] * target.shape[1]))
        
        err.update(err_batch, target.shape[0] * target.shape[1])
        
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()
        
    
        if i % args.print_freq == 0 and i != 0: 
            progress.display(i)
    return err.avg, losses.avg


def validate(val_loader, model, criterion, therehold, args, n=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    err = AverageMeter('Error', ':6.2f')
    err_q = AverageMeter('Error', ':6.2f')
    stat_target = None
    err_spe = None
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True).float()
            target = target.cuda(args.gpu, non_blocking=True).float()
            if args.arch == 'Q2L':
                pred = model(images)
            elif args.arch == 'L2L':
                pred_q, pred, _, _ = model(x=images)
                if args.head == 'r2r':
                    pred = target
            loss = criterion(pred, target)
            
            res = torch.where(pred > therehold, torch.ones_like(pred), torch.zeros_like(pred))
            stat_target = torch.sum(target.clamp(min=0), dim=0) if stat_target == None else stat_target + torch.sum(target.clamp(min=0), dim=0) 
            err_spe = torch.sum(res == target, dim=0).int() if err_spe == None else err_spe + torch.sum(res == target, dim=0).int() 
            err_batch = 100 - torch.sum(res == target) * (100 / (target.shape[0] * target.shape[1]))
            if loss != 0:
                losses.update(loss.item(), images.size(0))
            if args.arch == 'L2L':
                res_q = torch.where(pred_q > 0, torch.ones_like(pred_q), torch.zeros_like(pred_q))
                err_q_batch = 100 - torch.sum(res_q == target) * (100 / (target.shape[0] * target.shape[1]))
                err_q.update(err_q_batch, target.shape[0] * target.shape[1])
            err.update(err_batch, target.shape[0] * target.shape[1])
            batch_time.update(time.time() - end)
            end = time.time()
        err_spe = 100 - err_spe.float() * (100 / len(val_loader.dataset))
        stat_target = stat_target * (100 / len(val_loader.dataset))
        msg =  'Test: Error@1 {err.avg:.3f} QError@1 {qerr.avg:.3f} Loss {losses.avg:.3f} Time {batch_time.sum:.1f}'.format(err=err,qerr=err_q, losses=losses,batch_time=batch_time)
        logger.info(msg)
        return err.avg, losses.avg


from torchmetrics.functional import accuracy


def test(val_loader, model, criterion, therehold, args, n=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    err = AverageMeter('Error', ':6.2f')
    err_q = AverageMeter('Error', ':6.2f')
    stat_target = None
    err_spe = None
    all_batch_acc = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True).float()
            target = target.cuda(args.gpu, non_blocking=True).float()
            if args.arch == 'Q2L':
                pred = model(images)
            elif args.arch == 'L2L':
                pred_q, pred, _, _ = model(x=images)
                if args.head == 'r2r':
                    pred = target
            loss = criterion(pred, target)
            
            y_prob = torch.softmax(pred, dim=-1)
            y_prob = (y_prob > 0.5).float() 
            matric_acc =  accuracy(y_prob, target, task = 'multiclass', num_classes=3)
            print("test_acc", matric_acc)
            all_batch_acc.append(matric_acc.item()) 
            
            res = torch.where(pred > therehold, torch.ones_like(pred), torch.zeros_like(pred))
            stat_target = torch.sum(target.clamp(min=0), dim=0) if stat_target == None else stat_target + torch.sum(target.clamp(min=0), dim=0) 
            err_spe = torch.sum(res == target, dim=0).int() if err_spe == None else err_spe + torch.sum(res == target, dim=0).int() 
            err_batch = 100 - torch.sum(res == target) * (100 / (target.shape[0] * target.shape[1]))
            if loss != 0:
                losses.update(loss.item(), images.size(0))
            # if args.arch == 'L2L':
                res_q = torch.where(pred_q > 0, torch.ones_like(pred_q), torch.zeros_like(pred_q))
                err_q_batch = 100 - torch.sum(res_q == target) * (100 / (target.shape[0] * target.shape[1]))
                err_q.update(err_q_batch, target.shape[0] * target.shape[1])
            err.update(err_batch, target.shape[0] * target.shape[1])
            batch_time.update(time.time() - end)
            end = time.time()
        
        #최종 acc 파일에 저장
        final_acc = sum(all_batch_acc) / len(all_batch_acc) 
        filename = "/home/dang/experiment/Face_Attribute/result/origin_down_acc.csv"
        append_to_csv(filename , [final_acc])
        
        err_spe = 100 - err_spe.float() * (100 / len(val_loader.dataset))
        stat_target = stat_target * (100 / len(val_loader.dataset))
        msg =  'Test: Error@1 {err.avg:.3f} QError@1 {qerr.avg:.3f} Loss {losses.avg:.3f} Time {batch_time.sum:.1f}'.format(err=err,qerr=err_q, losses=losses,batch_time=batch_time)
        logger.info(msg)
        return err.avg, losses.avg


    
if __name__ == '__main__':
    args = get_argparse()
    main(args)
    
