from train import get_trainer
from options import prepare_train_args
from utils.utils import set_seed

def main():
    args = prepare_train_args()
    set_seed(args.seed)

    if args.k_fold > 0:
        for trainer in get_trainer(args):
            trainer.train()
    else:
        trainer = get_trainer(args)
        trainer.train()

if __name__ == '__main__':
    main()