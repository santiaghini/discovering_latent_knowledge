import os
import argparse

import numpy as np
import torch
# make sure to install promptsource, transformers, and datasets!
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM

# For COSMIC_QA, best is prompt idx 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############# Model loading and result saving #############

# Map each model name to its full Huggingface name; this is just for convenience for common models. You can run whatever model you'd like.
# TODO: Test with two or three, unifiedQA (best model from zero-shot)
# TODO: Three datasets
model_mapping = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-mnli": "microsoft/deberta-xxlarge-v2-mnli",
    "deberta": "microsoft/deberta-xxlarge-v2",
    "roberta-mnli": "roberta-large-mnli",
    "distilbert": "distilbert-base-uncased",
    "distilbert-race": "gsgoncalves/distilbert-base-uncased-race",
    "gpt2": "gpt2",
    "unifiedqa-t5-sm": "allenai/unifiedqa-t5-small",
    "unifiedqa-v2-sm": "allenai/unifiedqa-v2-t5-small-1363200",
    "gpt2-cosmos": "flax-community/gpt2-Cosmos",
    "roberta-race": "LIAMF-USP/roberta-large-finetuned-race",
    "bert-swag": "aaraki/bert-base-uncased-finetuned-swag",
    "t5-cosmos": "mamlong34/t5_small_cosmos_qa"
}


def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="T5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", type=str, default="cosmos_qa", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="train", help="Which split of the dataset to use")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Which prompt to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    # which hidden states we extract
    parser.add_argument("--use_decoder", action="store_true", help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states")

    return parser


def load_model(model_name, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    if model_name in model_mapping:
        # use a nickname for our models
        full_model_name = model_mapping[model_name]
    else:
        # if you're trying a new model, make sure it's the full name
        full_model_name = model_name

    # use the right automodel, and get the corresponding model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "decoder"
    
        
    # specify model_max_length (the max token length) to be 512 to ensure that padding works 
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type


def save_generations(generation, args, generation_type):
    """
    Input: 
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: one of "negative_hidden_states" or "positive_hidden_states" or "labels"

    Saves the generations to an appropriate directory.
    """
    # construct the filename based on the args
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)

    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save
    np.save(os.path.join(args.save_dir, filename), generation)


def load_single_generation(args, generation_type="hidden_states"):
    # use the same filename as in save_generations
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = generation_type + "__" + "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy".format(generation_type)
    return np.load(os.path.join(args.save_dir, filename))


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    c0_hs = load_single_generation(args, generation_type="c0_hidden_states")
    c1_hs = load_single_generation(args, generation_type="c1_hidden_states")
    c2_hs = load_single_generation(args, generation_type="c2_hidden_states")
    c3_hs = load_single_generation(args, generation_type="c3_hidden_states")
    labels = load_single_generation(args, generation_type="labels")

    return c0_hs, c1_hs, c2_hs, c3_hs, labels
