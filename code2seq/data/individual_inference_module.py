import os

class IndividualInferenceModule:

    def __init__(self):
        super().__init__()

        # TODO - 1 - Upload to ec2 instance and get preprocessed ast
        # TODO - 2 - Extract raw paths and pass into model - ✅
        # TODO - 3 - Convert ast_test.py into a class with proper method definition conventions so that a simple java file can be passed in and the outputs of the model can recieved out
        # TODO - 4 - Respond to Egor Spirin
        # TODO - 5 - experiment with taking only one sample from the batch for train.c2s dataset using data loaders which are defined now
        # TODO - 6 - get ast and build the vocabulary with it - ✅
        # TODO - 7 - create a new inference configuration with only configs needed for the dataloader
        # TODO - 8 - use get path method to get the ast paths - ✅
        # TODO - 9 - Retest model with preprocessed AST
        # TODO - 10 - Extract output of model from middle encoder layer