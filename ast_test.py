import os
import torch
from typing import List,Dict,Sequence,Optional
from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig,OmegaConf
from torch.utils.data import Dataset,DataLoader

from code2seq.data.path_context import LabeledPathContext,BatchedLabeledPathContext,Path
from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.path_context_data_module import PathContextDataModule
from code2seq.data.vocabulary import Vocabulary
from code2seq.model import Code2Seq
from commode_utils.vocabulary import build_from_scratch


# data_file = "val.c2s"
# # data_file = "train_ast.raw.txt"

# line_offsets = get_lines_offsets(data_file)
# raw_label, *raw_path_contexts = get_line_by_offset(data_file, line_offsets[0])

# max_context = 200
# n_contexts = min(len(raw_path_contexts), max_context)

# raw_paths = raw_path_contexts[:n_contexts]
# # paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]

# print(raw_paths)  
# TODO - 1 - Upload to ec2 instance and get preprocessed ast
# TODO - 2 - Extract raw paths and pass into model

def collate_wrapper(batch: List[Optional[LabeledPathContext]]) -> BatchedLabeledPathContext:
        return BatchedLabeledPathContext(batch)



data_dir = "generated_dataset/"
# data_file_path = os.path.join(data_dir,"train.c2s")
data_file_path = os.path.join(data_dir,"val.c2s")


config = OmegaConf.load("config/code2seq-java-small.yaml")

data_config = config.data
model_config = config.model

vocabulary_path = os.path.join(data_dir,Vocabulary.vocab_filename)

if not os.path.exists(vocabulary_path):
    build_from_scratch(data_file_path,Vocabulary)
_vocabulary = Vocabulary(vocabulary_path,data_config.max_labels,data_config.max_tokens,is_class=False)
dataset = PathContextDataset(data_file=data_file_path,config=data_config,vocabulary=_vocabulary,random_context=False)



print("DATASET LEN : ",len(dataset))
# print(len(dataset[0].path_contexts))
# print(dataset[1])
# print(dataset[2])

dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate_wrapper,pin_memory=True)
# dataloader = DataLoader(
#     dataset    


# model = 

# print("PATH  : ",paths)

# initializing model
code2seq_model = Code2Seq(model_config=config.model,optimizer_config=config.optimizer,vocabulary=_vocabulary,teacher_forcing=config.train.teacher_forcing)



# runnning inference by extracting only the first sample of in the val.c2s dataset 
dataloader_item = next(iter(dataloader))


# PRINT MODEL INPUTS
#############################3

print("Contexts Per Label : ",dataloader_item.contexts_per_label)
print("From Token : ",dataloader_item.from_token)
print("Path Nodes : ",dataloader_item.path_nodes)
print("To Token : ",dataloader_item.to_token)
print("Output Lenght : ",dataloader_item.labels.shape[0])

#######################


with torch.no_grad():
    model_output = code2seq_model(

        dataloader_item.from_token, # from_token
        dataloader_item.path_nodes, # path_nodes
        dataloader_item.to_token, # to_token
        dataloader_item.contexts_per_label, # contexts_per_label
        dataloader_item.labels.shape[0], # output size
        None # target_sequence  (since we are running in inference mode)
    )

    print("Model Output : ",model_output)












