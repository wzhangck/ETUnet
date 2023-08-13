import os
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from tqdm import tqdm
import time
import shutil
import torch
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import AverageMeter
import torch.utils.data.distributed
from torch.cuda.amp import autocast as autocast,GradScaler


def train_epoch(model,loader,optimizer,epoch,args,loss_func,scaler):

    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    iter = len(loader)

    for idx, batch in enumerate(loader):

        image = batch["image"].to(args.device)
        target = batch["label"].to(args.device)

        optimizer.zero_grad()

        with autocast(enabled=True):
            logits = model(image)
            loss = loss_func(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logits = model(image)

        # loss = loss_func(logits, target)
        # loss.backward()
        # optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, iter),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))

        start_time = time.time()

    return run_loss.avg

# 保存模型权重
def save_checkpoint(model,epoch,args,filename='model.pt',best_acc=0,optimizer=None,scheduler=None):
    state_dict = model.state_dict() # 保存模型参数
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)

    print('Saving checkpoint', filename)


class Trainer:
    def __init__(self, args,
                 train_loader,
                 loss_func,
                 validator=None,
                 ):
        pass
        self.args = args
        self.train_loader = train_loader
        self.validator = validator
        self.loss_func = loss_func
        self.scaler = GradScaler()

    def train(self, model,optimizer, scheduler=None,start_epoch=0,):
        pass
        args = self.args
        train_loader = self.train_loader
        writer = None

        if args.logdir is not None and args.rank == 0:
            writer = SummaryWriter(log_dir=args.logdir)

        val_acc_max_mean = 0.
        val_acc_max = 0.

        for epoch in range(start_epoch, args.max_epochs):

            epoch_time = time.time()
            train_loss = train_epoch(model,
                                     train_loader,
                                     optimizer,
                                     epoch=epoch,
                                     args=args,
                                     loss_func=self.loss_func,
                                     scaler=self.scaler,
                                     )

            if args.rank == 0:
                print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                      'time {:.2f}s'.format(time.time() - epoch_time))

            if args.rank==0 and writer is not None:
                writer.add_scalar('train_loss', train_loss, epoch)

            b_new_best = False

            if (epoch+1) % args.val_every == 0 and self.validator is not None:
                epoch_time = time.time()

                val_avg_acc = self.validator.run()

                mean_dice = self.validator.metric_dice_avg(val_avg_acc)
                mean_hd95 = self.validator.metric_hd95_avg(val_avg_acc)

                if args.rank == 0:
                    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                          'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time),
                          "mean_dice", mean_dice,'mean_hd95',mean_hd95)

                    if writer is not None:
                        for name, value in val_avg_acc.items():
                            if "dice" in name.lower():
                                writer.add_scalar(name, value, epoch)
                            if "hd95" in name.lower():
                                writer.add_scalar(name, value, epoch)
                                
                        writer.add_scalar('mean_dice', mean_dice, epoch)
                        writer.add_scalar('mean_hd95', mean_hd95, epoch)

                    if mean_dice > val_acc_max_mean:
                        print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max_mean, mean_dice))
                        val_acc_max_mean = mean_dice
                        val_acc_max = val_avg_acc
                        b_new_best = True

                        if args.rank == 0 and args.logdir is not None:
                            save_checkpoint(model, epoch, args,
                                            best_acc=val_acc_max_mean,
                                            optimizer=optimizer,
                                            scheduler=scheduler)

                if args.rank == 0 and args.logdir is not None:
                    with open(os.path.join(args.logdir, "log.txt"), "a+") as f:
                        f.write(f"epoch:{epoch+1}, metric:{val_avg_acc}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, avg metric: {mean_dice,mean_hd95}")
                        f.write("\n")
                        f.write(f"epoch:{epoch+1}, best dice metric:{val_acc_max}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, best avg dice metric: {val_acc_max_mean}")
                        f.write("\n")
                        f.write("*" * 20)
                        f.write("\n")

                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=val_acc_max,
                                    filename='model_final_best.pt'
                                    )

                    if b_new_best:
                        print('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.logdir, 'model_final_best.pt'), os.path.join(args.logdir, 'model.pt'))

            if scheduler is not None:
                scheduler.step()

        print('Training Finished !, Best Accuracy: ', val_acc_max)

        return val_acc_max

class Validator:
    def __init__(self,
                 args,
                 model,
                 val_loader,
                 class_list,
                 metric_functions,
                 sliding_window_infer=None,
                 post_label=None,
                 post_pred=None,
                 ) -> None:

        self.val_loader = val_loader
        self.sliding_window_infer = sliding_window_infer
        self.model = model
        self.args = args
        self.post_label = post_label
        self.post_pred = post_pred
        self.metric_functions = metric_functions
        self.class_list = class_list

    def metric_dice_avg(self, metric):
        metric_sum = 0.0
        c_nums = 0
        for m, v in metric.items():
            if "dice" in m.lower():
                metric_sum += v
                c_nums += 1

        return metric_sum / c_nums
    
    def metric_hd95_avg(self, metric):
        metric_sum = 0.0
        c_nums = 0
        for m, v in metric.items():
            if "hd95" in m.lower():
                metric_sum += v
                c_nums += 1

        return metric_sum / c_nums

    def is_best_metric(self, cur_metric, best_metric):

        best_metric_sum = self.metric_dice_avg(best_metric)
        metric_sum = self.metric_dice_avg(cur_metric)
        if best_metric_sum < metric_sum:
            return True

        return False

    def run(self):

        self.model.eval()
        args = self.args

        assert len(self.metric_functions[0]) == 2

        accs = [None for i in range(len(self.metric_functions))]
        not_nans = [None for i in range(len(self.metric_functions))]

        class_metric = []

        for m in self.metric_functions:
            for clas in self.class_list:
                class_metric.append(f"{clas}_{m[0]}")

        for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):

            val_image=batch["image"].to(args.device)
            val_label = batch["label"].to(args.device)

            with torch.no_grad():
                if self.sliding_window_infer is not None:
                    logits = self.sliding_window_infer(inputs=val_image,network=self.model)
                else:
                    logits = self.model(val_image)

                if self.post_label is not None:
                    val_label = self.post_label(val_label)

                if self.post_pred is not None:
                    logits = self.post_pred(logits)

                for i in range(len(self.metric_functions)): # 1

                    acc = self.metric_functions[i][1](y_pred=logits, y=val_label)
                    acc, not_nan = do_metric_reduction(acc, MetricReduction.MEAN_BATCH)

                    acc = acc.detach().cpu()  #
                    not_nan = not_nan.detach().cpu()

                    if accs[i] is None:

                        accs[i] = acc
                        not_nans[i] = not_nan
                    else :
                        accs[i] += acc
                        not_nans[i] += not_nan

        accs = torch.stack(accs, dim=0).flatten()
        not_nans = torch.stack(not_nans, dim=0).flatten()
        not_nans[not_nans == 0] = 1
        accs = accs / not_nans

        all_metric_list = {k: v for k, v in zip(class_metric, accs.tolist())}

        return all_metric_list
