#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
from datasets import load_dataset


glue_tasks = [
    "rte",
    "qqp",
    "sst2",
    "mrpc"
]

def get(logger,args):
    """
    load data for dataset_name
    """
    dataset_name = 'glue'
    classes = glue_tasks

    print('dataset_name: ',dataset_name)

    data={}
    taskcla=[]

    # Others
    f_name = dataset_name + '_random_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)

    dataset = {}
    dataset['train'] = {}
    dataset['valid'] = {}
    dataset['test'] = {}



    for c_id,cla in enumerate(classes):
        d = load_dataset(dataset_name,cla, split='train')
        d_split = d.train_test_split(test_size=0.2,shuffle=True,seed=args.seed)
        dataset['train'][c_id] = d_split['train']

        d_split = d_split['test'].train_test_split(test_size=0.5,shuffle=True,seed=args.seed) # test into half-half

        dataset['test'][c_id] = d_split['test']
        dataset['valid'][c_id] = d_split['train']
        # 50% of test for valid

    class_per_task = args.class_per_task

    examples = {}
    for s in ['train','test','valid']:
        examples[s] = {}
        for c_id, c_data in dataset[s].items():
            nn=(c_id//class_per_task) #which task_id this class belongs to

            if nn not in examples[s]: examples[s][nn] = []
            for c_dat in c_data:
                text= c_dat['text']
                label = c_id%class_per_task
                examples[s][nn].append((text,label))

    for t in range(args.ntasks):
        t_seq = int(random_sep[t].split('_')[-1])
        data[t]={}
        data[t]['ncla']=class_per_task
        data[t]['name']=dataset_name+'_'+str(t_seq)
        taskcla.append((t,int(data[t]['ncla'])))

        for s in ['train','test','valid']:
            if s == 'train':
                processor = data_utils.DtcProcessor()
                label_list = processor.get_labels(args.ntasks)

                tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
                train_examples =  processor._create_examples(examples[s][t_seq], "train")

                #TODO: in case you want to cut data, insert here

                num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
                # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

                train_features = data_utils.convert_examples_to_features_dtc(
                    train_examples, label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(train_examples))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)

                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)
                train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask,
                                           all_label_ids, all_tasks)

                data[t]['train'] = train_data
                data[t]['num_train_steps']=num_train_steps


            if s == 'valid':
                # processor = data_utils.DtcProcessor()
                # label_list = processor.get_labels(args.ntasks)
                # tokenizer = ABSATokenizer.from_pretrained(args.bert_model)

                valid_examples = processor._create_examples(examples[s][t_seq], "valid")
                valid_features=data_utils.convert_examples_to_features_dtc(
                    valid_examples, label_list, args.max_seq_length, tokenizer)

                valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
                valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
                valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
                valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
                valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

                valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                           valid_all_label_ids, valid_all_tasks)

                logger.info("***** Running validations *****")
                logger.info("  Num orig examples = %d", len(valid_examples))
                logger.info("  Num split examples = %d", len(valid_features))
                logger.info("  Batch size = %d", args.train_batch_size)

                data[t]['valid']=valid_data

            if s == 'test':
                processor = data_utils.DtcProcessor()
                label_list = processor.get_labels(args.ntasks)
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                eval_examples = processor._create_examples(examples[s][t_seq], "test")
                eval_features = data_utils.convert_examples_to_features_dtc(eval_examples, label_list, args.max_seq_length, tokenizer)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask,
                                          all_label_ids, all_tasks)
                # Run prediction for full data

                data[t]['test']=eval_data


    # total number of class
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla


