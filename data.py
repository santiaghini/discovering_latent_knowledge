import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from datasets import load_dataset

# Test SWAG
# TODO OH: choose the right dataset and model, making sure it performs well
#   bert (encoder) is not great for this. maybe use gpt-2 instead, decoder only models
datasets = {
    "ai2_arc" : {"subset_name": "ARC-Easy", "prompt_name": "raw_dataset", "label_key": "answer", "labels_set": ["A", "B", "C", "D"]},
    "race": {"subset_name": "all", "prompt_name": "Select the best answer", "label_key": "answer", "labels_set": ["A", "B", "C", "D"]},
    "swag": {"subset_name": "regular", "prompt_name": "how_ends", "label_key": "label", "labels_set": [0, 1, 2, 3]},
    "hellaswag": {"subset_name": "", "prompt_name": "how_ends", "label_key": "label", "labels_set": [0, 1, 2, 3]},
    "cosmos_qa": {"subset_name": "", "prompt_name": "description_context_question_answer_id", "label_key": "label", "labels_set": [0, 1, 2, 3]}
}

############# Data #############
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
        # prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        # TODO: can experiment with changing the prompts used in the dataset
        self.prompt = all_prompts[prompt_name]

        self.labels_map = None
        labels_set = datasets[self.dataset_name]["labels_set"]
        if type(labels_set[0]) != int:
            self.labels_map = {}
            for i, label in enumerate(labels_set):
                self.labels_map[label] = i

        # self.labels_set = set()
        # for ex in self.raw_dataset:
        #     self.labels_set.add(ex["label"])
        #     if len(self.labels_set) == 4:
        #         break

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
        
        # TODO: verify this operation is correct
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


    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        # TODO: check if what's question / answer for amazon dataset: is answer the full answer or just the idx too?
        # for race dataset answer = idx of answer (e.g. "0")
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

        # get the possible labels
        # label_list = [x for x in [data['answer0'], data['answer1'], data['answer2'], data['answer3']] if x != '']
        # label_list = self.prompt.get_answer_choices_list(data)
        # assert len(label_list) == 4, print("Make sure there are exacly four possible answers! Actual number of answers:", label_list)

        labels_set = datasets[self.dataset_name]["labels_set"]

        label_key = datasets[self.dataset_name]["label_key"]
        true_answer = data[label_key]
        if self.labels_map:
            true_answer = self.labels_map[true_answer]

        # reconvert to dataset format but with fake/candidate labels to create the contrast pair
        c0_example = data.copy()
        c0_example[label_key] = labels_set[0]
        c1_example = data.copy()
        c1_example[label_key] = labels_set[1]
        c2_example = data.copy()
        c2_example[label_key] = labels_set[2]
        c3_example = data.copy()
        c3_example[label_key] = labels_set[3]

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        c0_prompt, c1_prompt, c2_prompt, c3_prompt = self.prompt.apply(c0_example), self.prompt.apply(c1_example), self.prompt.apply(c2_example), self.prompt.apply(c3_example)

        # tokenize
        c0_ids, c1_ids, c2_ids, c3_ids = self.encode(c0_prompt), self.encode(c1_prompt), self.encode(c2_prompt), self.encode(c3_prompt),

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and self.model_type == "encoder_decoder":
            assert (c0_ids["decoder_input_ids"] - c1_ids["decoder_input_ids"] - c2_ids["decoder_input_ids"] - c3_ids["decoder_input_ids"]).sum() != 0, print("The decoder_input_ids for the contrast pairs are the same!", c0_ids, c1_ids, c2_ids, c3_ids)
        else:
            assert (c0_ids["input_ids"] - c1_ids["input_ids"] - c2_ids["input_ids"] - c3_ids["input_ids"]).sum() != 0, print("The input_ids for the contrast pairs are the same!", c0_ids, c1_ids, c2_ids, c3_ids)

        # return the tokenized inputs, the text prompts, and the true label
        return c0_ids, c1_ids, c2_ids, c3_ids, c0_prompt, c1_prompt, c2_prompt, c3_prompt, true_answer

    
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

    # remove examples that would be truncated (since this messes up contrast pairs)
    prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
    prompt = all_prompts[prompt_name_list[prompt_idx]]
    keep_idxs = []
    for idx in random_idxs:
        question, answer = prompt.apply(raw_dataset[int(idx)])
        input_text = question + " " + answer
        if len(tokenizer.encode(input_text, truncation=False)) < tokenizer.model_max_length - 2:  # include small margin to be conservative
            keep_idxs.append(idx)
            if len(keep_idxs) >= num_examples:
                break

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader

