from evaluate import get_evaluator
from options import prepare_test_args
from utils.utils import set_seed

def main():
    args = prepare_test_args()
    set_seed(args.seed)
    evaluator = get_evaluator(args)
    metrics = evaluator.evaluate()
    if args.model_type == 'Base':
        hi_dict, nt_summary = evaluator.record()
        return metrics, hi_dict, nt_summary
    elif args.model_type == 'MS':
        hi_dict, deg_hi_dict, nt_summary = evaluator.record()
        return metrics, deg_hi_dict, nt_summary
    else:
        hi_dict = evaluator.record()
        return metrics, hi_dict


if __name__ == '__main__':
    test_results = main()
    print(f'Metrics:{test_results[0]}')
    if len(test_results) > 2:
        print(f'Test for Normality:{test_results[2]}')