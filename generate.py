from data import get_dataloader
from hs import get_all_hidden_states
from utils import get_parser, load_model, save_generations

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    dataloader = get_dataloader(args.dataset_name, args.split, tokenizer, args.prompt_idx, batch_size=args.batch_size, 
                                num_examples=args.num_examples, model_type=model_type, use_decoder=args.use_decoder, device=args.device)

    # Get the hidden states and labels
    print("Generating hidden states")
    all_choices_hs, y = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers, 
                                              token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder)

    # Save the hidden states and labels
    print("Saving hidden states")
    for i in range(len(all_choices_hs)):
        save_generations(all_choices_hs[i][0], args, generation_type=f"c{i}_correct_hs")
        save_generations(all_choices_hs[i][1], args, generation_type=f"c{i}_incorrect_hs")
    save_generations(y, args, generation_type="labels")


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()
    main(args)
