from ccs import CCS
from utils import get_parser, load_all_generations

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

def main(args, generation_args):
    # Load hidden states and labels
    all_choices_hs, y = load_all_generations(generation_args)

    all_ccs = []
    # Run per each choice
    for i in range(len(all_choices_hs)):
        print(f"Choice {i} of {len(all_choices_hs)}:")
        correct_hs, incorrect_hs = all_choices_hs[i]

        # Make sure the shape is correct
        assert correct_hs.shape == incorrect_hs.shape
        correct_hs, incorrect_hs = correct_hs[..., -1], incorrect_hs[..., -1]  # take the last layer
        if correct_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
            correct_hs = correct_hs.squeeze(1)
            incorrect_hs = incorrect_hs.squeeze(1)

        # Create labels
        y_choice = np.zeros_like(y)
        y_choice[y == i] = 1 # 1 means that this choice is the answer for the initial question

        # Very simple train/test split (using the fact that the data is already shuffled)
        correct_hs_train, correct_hs_test = correct_hs[:len(correct_hs) // 4*3], correct_hs[len(correct_hs) // 4*3:]
        incorrect_hs_train, incorrect_hs_test = incorrect_hs[:len(incorrect_hs) // 4*3], incorrect_hs[len(incorrect_hs) // 4*3:]
        y_train, y_test = y_choice[:len(y_choice) // 4*3], y_choice[len(y_choice) // 4*3:]

        # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
        # you can also concatenate, but this works fine and is more comparable to CCS inputs
        if not args.skip_lr:
            x_train = correct_hs_train - incorrect_hs_train  
            x_test = correct_hs_test - incorrect_hs_test
            lr = LogisticRegression(class_weight="balanced", max_iter=args.lr_iter)
            lr.fit(x_train, y_train)
            print("\tLogistic regression accuracy: {}".format(lr.score(x_test, y_test)))

        # Set up CCS
        ccs = CCS(correct_hs_train, incorrect_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                        verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                        var_normalize=args.var_normalize)
        
        # train and evaluate CCS

        ccs.repeated_train()
        ccs_acc = ccs.get_acc(correct_hs_test, incorrect_hs_test, y_test)
        print(f"\tCCS accuracy for choice {i}: {ccs_acc}")

        all_ccs.append(ccs)
    
    # Integrating learned probes
    print("Now integrating and running altogether:")
    y_train, y_test = y[:len(y) // 4*3], y[len(y) // 4*3:]
    all_confidences = [ccs.avg_confidence for ccs in all_ccs]
    # stack the tensors into a new tensor
    stacked = torch.stack(all_confidences)
    # use the argmax function to find the index of the tensor with the largest value at each position
    predictions = stacked.argmax(dim=0)
    acc = (predictions[:, 0].cpu() == torch.tensor(y_test)).float().mean()
    print(f"Final multi-CCS accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = get_parser()
    generation_args, _ = parser.parse_known_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    parser.add_argument("--skip_lr", action="store_true")
    parser.add_argument("--lr_iter", type=int, default=1000)
    args, _ = parser.parse_known_args()
    main(args, generation_args)
