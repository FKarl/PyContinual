import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from apex import amp

import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append("./approaches/base/")
from bert_adapter_base import Appr as ApprBase
from my_optimization import BertAdam


class Appr(ApprBase):


    def __init__(self,model,logger, taskcla=None,args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('DIL BERT Adapter NCL')

        return

    def train(self,t,train,valid,num_train_steps,train_data,valid_data):

        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)


        best_loss=np.inf
        best_model=utils.get_model(self.model)

        epoch_runtimes = []

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*10*25))

            runtime = clock1 - clock0
            epoch_runtimes.append(runtime)
            print('Epoch runtime: ', runtime)

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        # calc avg runtime
        avg_runtime = np.mean(epoch_runtimes)
        print('Average runtime: ', avg_runtime)
        std_runtime = np.std(epoch_runtimes)
        print('Std runtime: ', std_runtime)


        # add data to the buffer
        print('len(train): ',len(train_data))
        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        print('samples_per_task: ',samples_per_task)

        loader = DataLoader(train_data, batch_size=samples_per_task)
        input_ids, segment_ids, input_mask, targets,_ = next(iter(loader))

        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        targets = targets.to(self.device)


        output_dict = self.model.forward(input_ids, segment_ids, input_mask)
        if 'dil' in self.args.scenario:
            cur_task_output=output_dict['y']
        elif 'til' in self.args.scenario:
            outputs=output_dict['y']
            cur_task_output = outputs[t]

        self.buffer.add_data(
            examples=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            logits = cur_task_output.data
        )


        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss=self.ce(output,targets)



            if not self.buffer.is_empty():
                buf_inputs, buf_labels,buf_logits, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)

                buf_task_inputs = buf_inputs.long()
                buf_task_segment = buf_segment_ids.long()
                buf_task_mask = buf_input_mask.long()
                buf_task_labels = buf_labels.long()
                buf_task_logits = buf_logits

                output_dict = self.model.forward(buf_task_inputs, buf_task_segment, buf_task_mask)
                if 'dil' in self.args.scenario:
                    cur_task_output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    cur_task_output = outputs[t]

                loss += self.args.beta * self.ce(cur_task_output, buf_task_labels)
                loss += self.args.alpha * self.mse(cur_task_output, buf_task_logits)



            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        return global_step

    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

                loss=self.ce(output,targets)

                _,pred=output.max(1)
                hits=(pred==targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss/total_num,total_acc/total_num,f1

