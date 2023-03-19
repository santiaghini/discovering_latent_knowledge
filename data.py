import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from datasets import load_dataset

############# Data #############

datasets = {
    "ai2_arc" : {"subset_name": "ARC-Easy", "prompt_name": "pick_the_most_correct_option", "label_key": "answerKey", "labels_set": [], "labels_map": {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}},
    "race": {"subset_name": "all", "prompt_name": "Select the best answer", "label_key": "answer", "labels_set": ["A", "B", "C", "D"], "labels_map": {"A": 0, "B": 1, "C": 2, "D": 3}},
    "swag": {"subset_name": "regular", "prompt_name": "how_ends", "label_key": "label", "labels_set": [0, 1, 2, 3], "labels_map": None},
    "hellaswag": {"subset_name": "", "prompt_name": "how_ends", "label_key": "label", "labels_set": [0, 1, 2, 3], "labels_map": None},
    "cosmos_qa": {"subset_name": "", "prompt_name": "description_context_question_answer_id", "label_key": "label", "labels_set": [0, 1, 2, 3], "labels_map": None}
}

class ContrastDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with a collection of prompts for that dataset from promptsource and a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(self, raw_dataset, tokenizer, all_prompts, dataset_name, prompt_idx, prompt_name,
                 model_type="encoder_decoder", use_decoder=False, device="cuda"):

        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.dataset_name = dataset_name
        self.all_prompts = all_prompts
        self.prompt_idx = prompt_idx
        self.prompt_name = prompt_name
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # for formatting the answers
        self.model_type = model_type
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert self.model_type != "encoder"

        # prompt
        self.prompt = all_prompts[prompt_name]

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt):
        """
        Tokenize a given natural language prompt (from after applying self.prompt to an example)
        
        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), or
        (2) feed the question the encoder and the answer to the decoder, creating contrast pairs using the decoder hidden states
        
        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt
        
        # tokenize the question and answer (depending upon the model type and whether self.use_decoder is True)
        if self.model_type == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer)
        elif self.model_type == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer)
        else:
            input_ids = self.get_decoder_input_ids(question, answer)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids


    def create_answer_hint(self, answer, correct):
        if correct:
            return f'\nOption "{answer}" is the correct answer'
        else:
            return f'\nOption "{answer}" is not the correct answer'


    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer 
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models.
        This is the same as get_encoder_input_ids except that we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_encoder_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use the encoder hidden states or the decoder hidden states.
        """
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer(answer, truncation=True, padding="max_length", return_tensors="pt")
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
        
        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids


    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]

        # want true_answer to be the index as int
        label_key = datasets[self.dataset_name]["label_key"]
        true_answer = data[label_key]
        labels_map = datasets[self.dataset_name]["labels_map"]
        if labels_map:
            true_answer = labels_map[true_answer]

        labels_set = datasets[self.dataset_name]["labels_set"]
        if self.dataset_name == "ai2_arc":
            labels_set = data["choices"]["label"]

        # reconvert to dataset format but with fake/candidate labels to create the contrast pair
        choice_objects = []
        for i in range(4):
            ex = data.copy()
            ex[label_key] = labels_set[i]
            question, answer = self.prompt.apply(ex)

            correct_prompt = (question, self.create_answer_hint(answer, True))
            incorrect_prompt = (question, self.create_answer_hint(answer, False))
            
            choice_objects.append({"prompts": [correct_prompt, incorrect_prompt], "encoded_ids": [self.encode(correct_prompt), self.encode(incorrect_prompt)]})


        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        enc_0 = choice_objects[0]["encoded_ids"]
        if self.use_decoder and self.model_type == "encoder_decoder":
            assert (enc_0[0]["decoder_input_ids"] - enc_0[1]["decoder_input_ids"]).sum() != 0, print("The decoder_input_ids for the contrast pairs are the same!")
        else:
            assert (enc_0[0]["input_ids"] - enc_0[1]["input_ids"]).sum() != 0, print("The input_ids for the contrast pairs are the same!")

        # return the tokenized inputs, the text prompts, and the true label
        return choice_objects, true_answer

    
def get_dataloader(dataset_name, split, tokenizer, prompt_idx, batch_size=16, num_examples=1000,
                   model_type="encoder_decoder", use_decoder=False, device="cuda", pin_memory=True, num_workers=1):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and prompt index

    Takes a random subset of (at most) num_examples samples from the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    subset_name = datasets[dataset_name]["subset_name"]
    if subset_name:
        raw_dataset = load_dataset(dataset_name, name=subset_name)[split]    
    else:
        raw_dataset = load_dataset(dataset_name)[split]

    # load all the prompts for that dataset
    all_prompts = DatasetTemplates(dataset_name, subset_name)

    prompt_name = datasets[dataset_name]["prompt_name"]

    # create the ConstrastDataset
    contrast_dataset = ContrastDataset(raw_dataset, tokenizer, all_prompts, dataset_name, prompt_idx, prompt_name,
                                       model_type=model_type, use_decoder=use_decoder, device=device)

    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    random_idxs = np.random.permutation(len(contrast_dataset))

    # remove examples that would be truncated (since this messes up contrast pairs) AND that have at most 4 choices
    prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
    prompt = all_prompts[prompt_name_list[prompt_idx]]
    keep_idxs = []
    for idx in random_idxs:
        question, answer = prompt.apply(raw_dataset[int(idx)])
        input_text = question + " " + answer
        if len(tokenizer.encode(input_text, truncation=False)) < tokenizer.model_max_length - 2:  # include small margin to be conservative
            if dataset_name == "ai2_arc" and len(raw_dataset[int(idx)]["choices"]["label"]) != 4:
                continue
            keep_idxs.append(idx)
            if len(keep_idxs) >= num_examples:
                break

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader