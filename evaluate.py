from ccs import CCS
from data import datasets
from utils import get_parser, load_all_generations

import numpy as np
from sklearn.linear_model import LogisticRegression

def main(args, generation_args):
    # Add key to args
    # generation_args.prompt_name = datasets[generation_args.dataset_name]["prompt_name"].replace(" ", "_")

    # load hidden states and labels
    print("Loading generations...")
    c0_hs, c1_hs, c2_hs, c3_hs, y = load_all_generations(generation_args)

    # Make sure the shape is correct
    assert c0_hs.shape == c1_hs.shape and c0_hs.shape == c2_hs.shape and c0_hs.shape == c3_hs.shape
    c0_hs, c1_hs, c2_hs, c3_hs = c0_hs[..., -1], c1_hs[..., -1], c2_hs[..., -1], c3_hs[..., -1]  # take the last layer
    if c0_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        c0_hs = c0_hs.squeeze(1)
        c1_hs = c1_hs.squeeze(1)
        c2_hs = c2_hs.squeeze(1)
        c3_hs = c3_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    c0_hs_train, c0_hs_test = c0_hs[:len(c0_hs) // 4*3], c0_hs[len(c0_hs) // 4*3:]
    c1_hs_train, c1_hs_test = c1_hs[:len(c1_hs) // 4*3], c1_hs[len(c1_hs) // 4*3:]
    c2_hs_train, c2_hs_test = c2_hs[:len(c2_hs) // 4*3], c2_hs[len(c2_hs) // 4*3:]
    c3_hs_train, c3_hs_test = c3_hs[:len(c3_hs) // 4*3], c3_hs[len(c3_hs) // 4*3:]
    y_train, y_test = y[:len(y) // 4*3], y[len(y) // 4*3:]

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs

    if not args.skip_lr:
        print("Starting logistic regression...")
        x_train = np.concatenate((c0_hs_train, c1_hs_train, c2_hs_train, c3_hs_train), axis=1)
        x_test = np.concatenate((c0_hs_test, c1_hs_test, c2_hs_test, c3_hs_test), axis=1)
        lr = LogisticRegression(class_weight="balanced", multi_class="multinomial", max_iter=args.lr_max_iter, penalty='l2', C=1.0)
        lr.fit(x_train, y_train)
        print("Logistic regression train accuracy: {}".format(lr.score(x_train, y_train)))
        print("Logistic regression test accuracy: {}".format(lr.score(x_test, y_test)))

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    ccs = CCS(c0_hs_train, c1_hs_train, c2_hs_train, c3_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize, info_loss_weight=args.info_loss_weight, cons_loss_weight=args.cons_loss_weight)
    
    # train and evaluate CCS
    print("Training...")
    ccs.repeated_train()
    print("Evaluating...")
    ccs_acc = ccs.get_acc(c0_hs_test, c1_hs_test, c2_hs_test, c3_hs_test, y_test)
    print("CCS accuracy: {}".format(ccs_acc))


if __name__ == "__main__":
    parser = get_parser()
    generation_args, _ = parser.parse_known_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--lr_max_iter", type=int, default=1000)
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
    parser.add_argument("--info_loss_weight", type=float, default=1.0)
    parser.add_argument("--cons_loss_weight", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    main(args, generation_args)
