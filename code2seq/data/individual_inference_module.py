import os
import torch
from typing import List,Dict,Optional
from omegaconf import DictConfig, OmegaConf
from code2seq.data.vocabulary import Vocabulary
from commode_utils.vocabulary import build_from_scratch
from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.path_context import LabeledPathContext,BatchedLabeledPathContext,Path
from code2seq.model import Code2Seq
from torch.utils.data import Dataset,DataLoader


class IndividualInferenceModule:

    def __init__(self,dataset_dir,config_path):
        
        self.data_dir = dataset_dir
        self.data_file_path = os.path.join(self.data_dir,"generated_ast_dataset.test.c2s")
        self.config = OmegaConf.load(config_path)
        self.data_config = self.config.data
        self.model_config = self.config.model
        self.optimizer_config = self.config.optimizer
        
        self.vocabulary_path = os.path.join(self.data_dir,Vocabulary.vocab_filename)
        self.vocabulary = None
    

    def run_inference(self,pretrained_model,print_params=False) : 

        """
        builds vocabulary and creates the dataset using the given .c2s data file. And then passes the dataset built into the model to get predictions.

        Params :
            pretrained_model : path to pretrained model
            print_params : If true prints the parameters of the dataset such as contexts per label and dataset len in addition to model architecture
 
        """        


        if not os.path.exists(self.vocabulary_path):
            build_from_scratch(self.data_file_path,Vocabulary)

        _vocabulary = Vocabulary(self.vocabulary_path,self.data_config.max_labels,
                                self.data_config.max_tokens,is_class=False)


        if _vocabulary is not None:
            dataset = PathContextDataset(
                    data_file=self.data_file_path,
                    config=self.data_config,
                    vocabulary=_vocabulary,
                    random_context=False)
        
        else:

            raise  Exception("Vocabulary is not built...")


        dataloader = DataLoader(dataset,batch_size=1,collate_fn=self.collate_wrapper,pin_memory=True)

        code2seq_model = Code2Seq(
                model_config=self.model_config,
                optimizer_config= self.optimizer_config,
                vocabulary = _vocabulary,
                teacher_forcing= self.config.train.teacher_forcing).load_from_checkpoint(pretrained_model)

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook


        # TODO - Analyze model architecture from code2seq research paper
        # TODO - Go deeper into the _encoder layer as well by referring the research paper and try to get outputs from encoder
        code2seq_model._encoder.register_forward_hook(get_activation("_encoder"))

        dataloader_item = next(iter(dataloader))

        with torch.no_grad():
            
            model_output = code2seq_model(
                    dataloader_item.from_token, # from_token
                    dataloader_item.path_nodes, # path_nodes
                    dataloader_item.to_token, # to_token
                    dataloader_item.contexts_per_label, # contexts_per_label
                    dataloader_item.labels.shape[0], # output size
                    None # target_sequence  (since we are running in inference mode)
            )

        if print_params:

            print("Dataset Len : ",len(dataloader))
            print("Contexts Per Label : ",dataloader_item.contexts_per_label)
            print("From Token : ",dataloader_item.from_token)
            print("Path Nodes : ",dataloader_item.path_nodes)
            print("To Token : ",dataloader_item.to_token)
            print("Output Lenght : ",dataloader_item.labels.shape[0])

            print("\nModel Architecture : ", code2seq_model)
            print("Model output shape : ",model_output.shape)
            print("Encoder Layer Activation : ",activation["_encoder"])

        output = {}
        output["model_output"] = model_output
        output["encoder_layer_output"] = activation["_encoder"] # TODO ADD OUTPUT OF MIDDLE LAYERS FROM ENCODERS AS WELL

        return output


    @staticmethod
    def collate_wrapper(batch: List[Optional[LabeledPathContext]]) -> BatchedLabeledPathContext:
        return BatchedLabeledPathContext(batch)




# TODO - 1 - Upload to ec2 instance and get preprocessed ast  -✅
# TODO - 2 - Extract raw paths and pass into model - ✅
# TODO - 3 - Convert ast_test.py into a class with proper method definition conventions so that a simple .c2s file can be passed in and the outputs of the model can recieved out - ✅
# TODO - 4 - Respond to Egor Spirin
# TODO - 5 - experiment with taking only one sample from the batch for train.c2s dataset using data loaders which are defined now
# TODO - 6 - get ast and build the vocabulary with it - ✅
# TODO - 7 - create a new inference configuration with only configs needed for the dataloader
# TODO - 8 - use get path method to get the ast paths - ✅
# TODO - 9 - Retest model with preprocessed AST - ✅
# TODO - 10 - Extract output of model from middle encoder layer - 
# TODO - 11 - Extract output from preprocessing file
