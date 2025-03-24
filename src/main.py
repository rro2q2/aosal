import torch
from model.bert import BertModel
from trainer import Trainer
import wandb
import os
from wandb.sdk.wandb_run import Run
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import time

from data.data_preprocessor import DataPreprocessor, MainDataset, print_dataset_statistics
from strategy.get_strategy import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # Load dataset 
    dataset = DataPreprocessor(
        data_dir=cfg.data_dir,
        train_test_split=cfg.train_test_split,
        ind_dataset=cfg.ind_dataset,
        ind_ratio=cfg.ind_ratio,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        max_seq_length=cfg.max_seq_length,
        ood_dataset=cfg.ood_dataset,
        ood_ratio=cfg.ood_ratio,
        noise_ratio=cfg.noise_ratio
    )

    # Prepare dataset and DataLoaders
    main_dataset, val_data_idxs, test_data_idxs = dataset.prepare_dataset()
    train_dl = dataset.train_dataloader(main_dataset)
    val_dl = dataset.val_dataloader(main_dataset, val_data_idxs)
    test_dl = dataset.test_dataloader(main_dataset, test_data_idxs)
    
    if dataset.ood_dataset or dataset.ood_samples: # If ODO dataset is present
        val_ood_dl = dataset.ood_val_dataloader(main_dataset, val_data_idxs)
    
    print_dataset_statistics(dataset)

    # Load model
    num_labels = len(dataset.labels)
    model = BertModel(num_labels=num_labels)

    # Load strategy
    strategy = get_sampling_strategy(cfg, model)

    # Load trainer
    trainer = Trainer(
        model=model, 
        learning_rate=cfg.learning_rate, 
        epochs=cfg.epochs, 
        weight_decay=cfg.weight_decay, 
        distance=cfg.distance, 
        percentile=cfg.percentile, 
        const_threshold=cfg.const_threshold
    )
    # Train and validate using initial labeled dataset
    trainer.train(train_dl, val_dl)
    
    run_name = f"{strategy.name.upper()}_{dataset.ind_dataset.upper()}"
    if dataset.ood_dataset:
        run_name += f"_{dataset.ood_dataset.upper()}_{dataset.noise_ratio}"   

    if cfg.strategy == 'aosal':
        aosal_map = {
            "mahalanobis": "MAH",
            "wasserstein": "WAS",
            "uncertainty": "UNC",
            "diversity": "DIV"
        }
        run_name += f"_{aosal_map[cfg.distance]}_{aosal_map[cfg.inf_measure]}_FPR{cfg.percentile}"
    
    ####### Initialize AL variables #######
    unlabeled_len = len(dataset.unlabeled_idxs)
    budget = int(strategy.budget_percent * unlabeled_len)
    print(f"Oracle budget size: {budget}")
    acq_size = int(strategy.acquisition_percent * unlabeled_len)
    print(f"Acquisition Size: {acq_size}")
    total_acq_size = 0

    # Start a new wandb run to track this script
    run = wandb.init(
        # Set the wandb project where this run will be logged
        project="al-ood-tods",
        name=run_name,
        # Track hyperparameters and run metadata
        config={
            "ind_dataset": cfg.ind_dataset,
            "ind_ratio": cfg.ind_ratio,
            "learning_rate": cfg.learning_rate,
            "architecture": "BERT",
            "epochs": cfg.epochs,
            "al_strategy": strategy.name,
        }
    )

    if dataset.ood_dataset:
        run.config["ood_dataset"] = cfg.ood_dataset
        run.config["ood_ratio"] = cfg.ood_ratio
        run.config["noise_ratio"] = cfg.noise_ratio

    if strategy.name == "cal":
        k = cfg.k
        run.config["strategy@k"] = k

    if strategy.name == "aosal":
        run.config["distance"] = cfg.distance
        run.config["fpr"] = cfg.percentile
        run.config["const_threshold"] = cfg.const_threshold 

    idx_rnd = 0
    print(f"############ Round {idx_rnd} ############")
    # Evaluate on test set
    test_results = trainer.test(test_dl)
    print(f"IND Test Loss: {test_results['avg_loss']:.3f} | IND Test Accuracy: {test_results['avg_acc']:.3f}")
    
    threshold = None
    if dataset.ood_dataset:
        # Test on IND/OOD validation set
        ood_val_results, threshold = trainer.evaluate_ood(train_dl, val_ood_dl)

    # Log initial results to WanDB
    run.log({
        "ind_test_acc": test_results["avg_acc"], 
        "ind_test_loss": test_results["avg_loss"], 
        "ood_val_acc": ood_val_results["avg_ood_acc"],
        "avg_ind": 0,
        "avg_ood": 0,
        "bgt_pcnt": 0
    })

    # active learning metrics
    budget_percentages = []
    acq_ratio = dict()
    test_accuracies = []
    test_losses = []

    # Perform AL
    idx_rnd += 1
    current_threshold = threshold
    start_time = time.time()
    while total_acq_size < budget:
        print(f"############ Round {idx_rnd} ############")
        b = acq_size # if (acq_size + total_acq_size) <= budget else (budget - total_acq_size)
        # perform query strategy
        query_idxs = strategy.query(b, main_dataset, dataset.train_data_idxs, dataset.unlabeled_idxs, current_threshold)
        print("Number of labeled samples:")
        print(len([idx for idx in query_idxs if dataset.labeled_samples[idx]]))
        # Update the total acquisition size
        total_acq_size += b
        # update IND/OOD% scores
        ind_query_idxs = [idx for idx in query_idxs if main_dataset[idx][1].item() != 999]
        ind_batch_percentage = float(len(ind_query_idxs) / b)
        ood_query_idxs = [idx for idx in query_idxs if main_dataset[idx][1].item() == 999]
        ood_batch_percentage =  float(len(ood_query_idxs) / b) #1 - ind_batch_percentage
        bgt_pcnt = total_acq_size / unlabeled_len
        acq_ratio[bgt_pcnt] = (len(ind_query_idxs), len(ood_query_idxs))
        budget_percentages.append(bgt_pcnt)

        print("---- Total Acquisition Size: {} | Current Acquisition Size: {} | Budget Percent: {:.3f} ----\n\
              Num IND Samples: {}\n\
              Num OOD Samples: {}\n\
              IND batch percentage: {:.3f}\n\
              OOD batch percentage: {:.3f}\n"
            .format(
                total_acq_size, 
                b, 
                bgt_pcnt, 
                acq_ratio[bgt_pcnt][0], 
                acq_ratio[bgt_pcnt][1], 
                ind_batch_percentage, 
                ood_batch_percentage
            )
        )
        # update IND and OOD labels
        new_train_dl = strategy.update_train_set(dataset, np.array(ind_query_idxs), main_dataset)
        new_val_ood_dl = strategy.update_val_ood_set(dataset, np.array(ood_query_idxs), main_dataset, val_data_idxs)
        
        # update current epoch
        trainer.current_epoch = trainer.epochs
        # train model
        trainer.train(new_train_dl, val_dl)
        # evaluate on test set (IND)
        test_results = trainer.test(test_dl)
        print(f"IND Test Loss: {test_results['avg_loss']:.3f} | IND Test Accuracy: {test_results['avg_acc']:.3f}") 

        if dataset.ood_dataset:
            # Test on IND/OOD validation set
            ood_val_results, threshold = trainer.evaluate_ood(train_dl, val_ood_dl)
            current_threshold = threshold

        avg_ind = sum([acq_ratio[idx][0] for idx in acq_ratio]) / len(acq_ratio)
        avg_ood = sum([acq_ratio[idx][1] for idx in acq_ratio]) / len(acq_ratio)
        print("Average IND samples per iteration: {:.3f}".format(avg_ind))
        print("Average OOD samples per iteration: {:.3f}".format(avg_ood))

        # Update metrics on WandDB
        run.log({
            "ind_test_acc": test_results["avg_acc"], 
            "ind_test_loss": test_results["avg_loss"], 
            "ood_val_acc": ood_val_results["avg_ood_acc"],
            "avg_ind": avg_ind,
            "avg_ood": avg_ood,
            "bgt_pcnt": int(bgt_pcnt*100)
        })

        idx_rnd += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"AL strategy '{cfg.strategy}' took {elapsed_time:.6f} seconds to execute.")
    # finish Wandb run
    run.finish()

if __name__ == '__main__':
    main()
