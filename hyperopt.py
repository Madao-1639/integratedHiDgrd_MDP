import optuna
from train import get_trainer
from options import prepare_train_args
from utils.utils import set_seed
from utils.preprocessing import read_data, gen_loo_data



def gen_hp(trial,args):
    # Generate hyperparams
        # NAS (Neural Architecture Search)
    # n_layers = trial.suggest_int('num_hidden_layers', 1, 2)
    n_layers = 2
    hidden_sizes = [trial.suggest_int(f'hidden_size_{layer}', 3, 100, log = True) for layer in range(1,n_layers+1)]
        # Other hyperparams
    params = {
        # 'dnn_hidden_sizes': hidden_sizes,
        # 'lstm_hidden_size':trial.suggest_int('lstm_hidden_size', 8, 64, log = True),
        'lstm_hidden_sizes': hidden_sizes,
        # 'mfe_loss_weight': trial.suggest_float('mfe_loss_weight', 1e-3, 1e-1, log = True),
        # 'mvf_loss_weight': trial.suggest_float('mvf_loss_weight', 0.8, 1.2),
        # 'mon_loss_weight': trial.suggest_float('mon_loss_weight', 0.1, 0.8),
        # 'con_loss_weight': trial.suggest_float('con_loss_weight', 0.01, 0.5, log = True),

        # Special hyperparams
        # 'MS_fix_point':trial.suggest_int('fix_point', 70, 100),
    }
    # Set new hyperparams to args
    for key, value in params.items():
        setattr(args, key, value)

def objective(trial):
    global args, data
    obj_metric = 'F1'
    gen_hp(trial,args)
    # Get trainer
    trainer = get_trainer(args,data)
    # Train & Val & Report
    # best_obj = 0
    for epoch in range(1,args.num_epoch+1):
        trainer.train_per_epoch(epoch)
        metrics = trainer.val_per_epoch(epoch)
        obj = metrics[obj_metric]
        trial.report(obj, epoch)
        # best_obj = max(best_obj,obj)
        if trial.should_prune():
            raise optuna.TrialPruned()
            # return best_obj   # Early stopping
    return obj
    # return best_obj

def objective_cv(trial):
    global args, data
    obj_metric = 'F1'
    gen_hp(trial,args)
    trainer_seq = get_trainer(args,data)
    best_obj = 0
    for epoch in range(1,args.num_epoch+1):
        obj_list = []
        for trainer in trainer_seq:
            trainer.train_per_epoch(epoch)
            metrics = trainer.val_per_epoch(epoch)
            obj_list.append(metrics[obj_metric])
        obj = sum(obj_list)/len(obj_list)
        trial.report(obj, epoch)
        best_obj = max(best_obj,obj)
        if trial.should_prune():
            # raise optuna.TrialPruned()
            return best_obj   # Early stopping
    # return obj
    return best_obj



args = prepare_train_args()
set_seed(args.seed)
study = optuna.create_study(
    study_name=args.model_name,
    storage='sqlite:///hyperopt.db',
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=args.seed), #  Specify a seed to make the parameters reproducible
    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=0), # Early stopping & prune
    # pruner=optuna.pruners.MedianPruner(),
    load_if_exists=True,
)
data = read_data(args.data_fp, args.drop_vars)
if 0 < args.test_ratio < 1:
    data, test_data = gen_loo_data(data,args.test_ratio)
    args.test_ratio = 0
if args.k_fold > 0:
    study.optimize(objective_cv, n_trials=200)
else:
    study.optimize(objective, n_trials=200)
