#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
import traceback
import csv # Added for CSV Logging
import datetime
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import wandb

DEFAULT_WORK_DIR = './work_dir/temp'
RESULT_COLUMNS = ['Timestamp', 'Experiment', 'Epoch', 'Top1_Acc',
                  'Top5_Acc', 'Top5_Policy', 'WorkDir']
TOP5_POLICY = 'same_epoch_as_logged_top1'

def init_seed(seed):
    # Multi-seed support: the previous implementation hard-coded seed=1 and
    # ignored the argument. We now honor the caller-supplied seed so that
    # tools/run_multiseed.py can produce 3-seed mean +/- std benchmarks.
    seed = int(seed) if seed is not None else 1
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default=DEFAULT_WORK_DIR,
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--train-fraction', type=float, default=1.0,
        help='fraction of the TRAIN set to keep (data-efficiency curve); '
             'injected into train_feeder_args as data_fraction when <1.0')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--wandb', default=True)
    parser.add_argument('--wandb_name', default='test')
    parser.add_argument('--wandb_entity', default='irvl')
    parser.add_argument('--wandb_project', default='SLGTformer First Run')
    parser.add_argument(
        '--timestamp-run',
        type=str2bool,
        default=False,
        help='append a timestamp to the resolved work directory')
    parser.add_argument(
        '--results-csv',
        default='results_final.csv',
        help='CSV file used for experiment metrics')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        self.arg = arg
        self.save_arg()

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

        if self.arg.wandb:
            wandb.init(name=self.arg.wandb_name, entity=self.arg.wandb_entity, project=self.arg.wandb_project, config=self.arg)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        # Data-efficiency curve: keep only a fraction of the train set.
        if float(getattr(self.arg, 'train_fraction', 1.0)) < 1.0:
            self.arg.train_feeder_args['data_fraction'] = float(self.arg.train_fraction)
            self.arg.train_feeder_args['fraction_seed'] = 42
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * len(self.arg.device),
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker * len(self.arg.device),
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print("Model Args:", self.arg.model_args)
        self.model = Model(**self.arg.model_args).to(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().to(output_device)
        # self.loss = LabelSmoothingCrossEntropy().to(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                ckpt = torch.load(self.arg.weights)
                if 'weights' in ckpt.keys():
                    weights = torch.load(self.arg.weights)['weights']
                else:
                    weights = ckpt

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.to(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = self.arg.weight_decay

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        if self.arg.weights:
            ckpt = torch.load(self.arg.weights)
            if 'optimizer' in ckpt.keys():
                opt_state_dict = ckpt['optimizer']
                self.optimizer.load_state_dict(opt_state_dict)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'eval_results'), exist_ok=True)

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if epoch >= self.arg.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = data.float().to(self.output_device)
            label = label.long().to(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate
            output = self.model(data, keep_prob)

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

            self.lr = self.optimizer.param_groups[0]['lr']

            if self.arg.wandb: wandb.log({
                'train_loss': loss.item(),
                'lr': self.lr,
            })

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in state_dict.items()])

        if save_model: 
            save_dict = {
                'weights': weights,
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr,
            }
            torch.save(save_dict, self.arg.model_saved_name +
                   '-' + str(epoch) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        with torch.no_grad():
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                process = tqdm(self.data_loader[ln])

                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)

                    with torch.no_grad():
                        output = self.model(data)

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' +
                                        str(x) + ',' + str(true[i]) + '\n')
                score = np.concatenate(score_frag)

                if 'UCLA' in self.arg.Experiment_name:
                    self.data_loader[ln].dataset.sample_name = np.arange(
                        len(score))

                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                top5_accuracy = self.data_loader[ln].dataset.top_k(score, 5)
                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                if accuracy > self.best_acc:
                    self.best_acc = accuracy

                    with open(os.path.join(self.arg.work_dir, 'eval_results', 'best_acc.pkl'), 'wb') as f:
                        pickle.dump(score_dict, f)

                    # Also persist the best MODEL (loadable via --weights). Per-epoch
                    # checkpoints follow save_interval and can miss the best epoch, so
                    # a run's peak model was otherwise unrecoverable.
                    best_weights = OrderedDict(
                        [[k.split('module.')[-1], v.cpu()]
                         for k, v in self.model.state_dict().items()])
                    torch.save({'weights': best_weights, 'epoch': epoch, 'acc': accuracy},
                               self.arg.model_saved_name + '-best.pt')

                print('Eval Accuracy: ', accuracy,
                    ' model: ', self.arg.model_saved_name)

                # ------------------------------------------------------------------
                # NEW: Log Best Result to CSV
                # ------------------------------------------------------------------
                csv_path = self.arg.results_csv
                append_result_row(csv_path, {
                    'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Experiment': self.arg.Experiment_name,
                    'Epoch': epoch,
                    'Top1_Acc': f"{accuracy:.4f}",
                    'Top5_Acc': f"{top5_accuracy:.4f}",
                    'Top5_Policy': TOP5_POLICY,
                    'WorkDir': self.arg.work_dir,
                })
                print(f"Logged best accuracy to {csv_path}")
                # ------------------------------------------------------------------

                if self.arg.wandb:  wandb.log({
                    'eval_accuracy': accuracy,
                    'eval_loss': np.mean(loss_value),
                })

                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

                if save_score:
                    score_name = 'epoch_{}_top1_{:.6f}.pkl'.format(epoch, accuracy)
                    with open(os.path.join(self.arg.work_dir, 'eval_results', score_name), 'wb') as f:
                        pickle.dump(score_dict, f)
        return np.mean(loss_value)
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            # Auto-create save directory
            import os
            if os.path.dirname(self.arg.model_saved_name):
                 os.makedirs(os.path.dirname(self.arg.model_saved_name), exist_ok=True)

            self.global_step = int(self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                should_eval = ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)
                if should_eval:
                    val_loss = self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        loader_name=['test'])

            print('best accuracy: ', self.best_acc,
                  ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args.get('debug', False):
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=self.arg.start_epoch, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def resolve_run_paths(arg):
    exp_name = arg.Experiment_name or os.path.splitext(os.path.basename(arg.config))[0] or 'temp'
    arg.Experiment_name = exp_name

    configured_work_dir = arg.work_dir
    if not configured_work_dir or configured_work_dir == DEFAULT_WORK_DIR:
        configured_work_dir = os.path.join('.', 'work_dir', exp_name)

    work_dir = configured_work_dir.rstrip('/\\')
    timestamped = False
    if arg.timestamp_run and arg.phase == 'train':
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = f"{work_dir}_{timestamp}"
        timestamped = True

    original_model_saved_name = arg.model_saved_name
    if not original_model_saved_name:
        model_saved_name = os.path.join(work_dir, f'{exp_name}_model')
    elif timestamped and os.path.normpath(original_model_saved_name).startswith(os.path.normpath(configured_work_dir)):
        model_saved_name = os.path.join(work_dir, os.path.basename(original_model_saved_name))
    else:
        model_saved_name = original_model_saved_name

    arg.work_dir = os.path.normpath(work_dir)
    arg.model_saved_name = os.path.normpath(model_saved_name)
    return arg


def append_result_row(csv_path, row):
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []

        if existing_fields != RESULT_COLUMNS:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
                writer.writeheader()
                for existing in existing_rows:
                    upgraded = {key: existing.get(key, '') for key in RESULT_COLUMNS}
                    if not upgraded['Top5_Policy']:
                        upgraded['Top5_Policy'] = TOP5_POLICY
                    writer.writerow(upgraded)

    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    
    arg = resolve_run_paths(arg)
    print(f"Work Dir: {arg.work_dir}")
    print(f"Model Save Prefix: {arg.model_saved_name}")

    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
