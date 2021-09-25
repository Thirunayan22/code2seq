from code2seq.data.individual_inference_module import IndividualInferenceModule



if __name__ == "__main__":

    dataset_dir = "generated_dataset"
    config_path = "config/code2seq-java-small.yaml"
    pretrained_model = "pretrained_model/pretrained_code2seq.ckpt"
    individual_inference = IndividualInferenceModule(dataset_dir=dataset_dir,
                                                    config_path=config_path)

    
    model_output = individual_inference.run_inference(pretrained_model,print_params=True)

    print("Model Output : " ,model_output)



