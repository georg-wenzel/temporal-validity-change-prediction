import os
import pickle
import random
from typing import List, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.display import clear_output


def stratified_target_statement_split(target_statements: List[str], val_fraction: float) -> Tuple[List[int], List[int]]:
    """
    Splits a list of target statements into a training and validation set, ensuring that target statements are
    unique in both sets. The split is stratified, i.e. all instances of a single target statement are part of
    either the training or the validation set.
    :param target_statements: A list of target statements as they appear in the PyTorch dataset.
    :param val_fraction: The fraction of the dataset to use for validation.
    :return: A tuple of two lists, the first containing the indices of the training set, the second containing the
    indices of the validation set.
    """
    # sort target statements by text
    sorted_indices = sorted(range(len(target_statements)), key=lambda k: target_statements[k])
    # each target statement should consist of 3 instances (one for each follow-up)
    # So we grab the closest number divisible by 3 of the val_fraction * len(target_statements)
    num_val_samples = int(np.floor(len(target_statements) * val_fraction / 3) * 3)
    # Split the sorted_indices and return
    return sorted_indices[num_val_samples:], sorted_indices[:num_val_samples]


def __collate_item(batch: List[dict], key: str, input_padding: int) -> torch.Tensor:
    """
    Collates a single item from a batch of data.

    :param batch: The batch to collate.
    :param key: The key of the item to collate.
    :param input_padding: The padding value to use for input_ids.
    :return: The collated item.
    """
    if key.startswith("labels"):
        # creates a tensor of shape (num_samples,) from a list of scalars (each label)
        return torch.stack([item[key] for item in batch])
    elif key.startswith("input_ids"):
        # creates a tensor of shape (num_samples, max_length) from a list of tensors (each input), using the input
        # padding value
        return torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True,
                                               padding_value=input_padding)
    else:
        # same stacking process, but padding value is default (0)
        return torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True)


def collate_batch(batch: List[dict], input_padding: int) -> dict:
    """
    Collates a batch of data into a single dictionary, where inputs are padded and stacked, and outputs are stacked.
    An output is identified by starting with 'labels' (e.g. 'labels_duration_standalone' or 'labels_change').

    :param batch: The batch to collate.
    :param input_padding: The padding value to use for input_ids.
    :return: The collated batch.
    """
    # return a dictionary of collated dimensions
    return {k: __collate_item(batch, k, input_padding) for k in batch[0].keys()}


def self_explain_collate_batch(batch: List[dict], input_padding: int) -> dict:
    """
    Pads a batch of samples to the same length, adds span information, and returns the collated batch.
    Based on https://github.com/ShannonAI/Self_Explaining_Structures_Improve_NLP_Models

    :param batch: The batch to collate.
    :param input_padding: The padding value to use for input_ids.
    :return: The collated batch.
    """
    # Collate the batch
    collated = {k: __collate_item(batch, k, input_padding) for k in batch[0].keys()}

    # generate all possible spans, excluding guaranteed invalid spans (i=0, j=max_len-1)
    max_len = max([len(item["input_ids"]) for item in batch])
    start_indices = []
    end_indices = []
    for i in range(1, max_len - 1):
        for j in range(i, max_len - 1):
            start_indices.append(i)
            end_indices.append(j)

    # generate span masks
    span_masks = []
    # for each sample
    for item in batch:
        span_mask = []
        # find the index of the </s> token in RoBERTa
        middle_index = item["input_ids"].tolist().index(2)
        # find the number of tokens in the input
        num_tokens = len(item["input_ids"])
        for start_index, end_index in zip(start_indices, end_indices):
            # if the span is not valid, mask it
            # A span is not valid if:
            # 1. the span crosses the sentence boundary (2 tokens: </s><s>)
            # 2. The span contains special tokens (i.e., end_index >= num_tokens - 1)
            if (start_index > middle_index + 1 or end_index < middle_index) and (end_index < num_tokens - 1):
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        span_masks.append(span_mask)
    # add the indices and span masks to the output
    collated["start_indices"] = torch.LongTensor(start_indices)
    collated["end_indices"] = torch.LongTensor(end_indices)
    collated["span_masks"] = torch.LongTensor(span_masks)
    return collated


def calculate_accuracy(outputs: List[torch.tensor], labels: List[torch.tensor]) -> float:
    """
    Calculates the accuracy of the model's predictions.

    :param outputs: The model's outputs.
    :param labels: The labels.
    :return: The accuracy.
    """
    outputs = torch.cat(outputs).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    # Calculate the accuracy
    accuracy = sum(np.argmax(outputs, axis=1) == labels) / len(labels)
    return accuracy


def calculate_em(inputs: List[torch.tensor], outputs: List[torch.tensor], labels: List[torch.tensor]) -> float:
    """
    Calculates the exact match score of the model's predictions.

    :param inputs: The model's inputs (Input IDs) as a numpy array.
    :param outputs: The model's outputs (logits) as a numpy array.
    :param labels: The labels as a numpy array.
    :return: The exact match score.
    """

    # Split the inputs, outputs, and labels into single sequences
    inputs = [idx.squeeze() for batch in inputs for idx in batch.cpu().split(1, dim=0)]
    outputs = [logits.squeeze() for batch in outputs for logits in batch.cpu().split(1, dim=0)]
    labels = [label.squeeze() for batch in labels for label in batch.cpu().split(1, dim=0)]

    # Check that the lengths are equal
    assert (len(inputs) == len(outputs) == len(labels))

    # Sort outputs and labels by input length
    sorted_indices = sorted(range(len(inputs)), key=lambda k: inputs[k].tolist())
    outputs_sorted = np.array(outputs)[sorted_indices]
    labels_sorted = np.array(labels)[sorted_indices]

    # Generate a sorted array of inputs
    max_dim = max([len(input) for input in inputs])
    inputs_sorted = np.zeros((len(inputs), max_dim))
    for i, item_at in enumerate(sorted_indices):
        inputs_sorted[i, :len(inputs[item_at])] = inputs[item_at]

    em = 0
    for i in range(0, len(sorted_indices), 3):
        # Find the separator token
        has_roberta_sep = len(np.where(inputs_sorted[i] == 2)[0])
        sep_index = np.where(
            inputs_sorted[i] == 2)[0][0] if has_roberta_sep else np.where(inputs_sorted[i] == 102)[0][0]
        # Get the subarray
        inputs_sub = inputs_sorted[i: i + 3, :sep_index]
        # Check if all three rows are equal
        assert (np.all(inputs_sub[0, :] == inputs_sub[1, :])
                and np.array_equal(inputs_sub[1, :], inputs_sub[2, :]))
        # If all three predictions are correct, add 1 to the EM score
        if np.array_equal(np.argmax(outputs_sorted[i:i + 3], axis=1), labels_sorted[i:i + 3]):
            em += 1
    # Return the EM score
    return em / (len(sorted_indices) / 3)


def set_up_deterministic_environment(seed: int) -> None:
    """
    Sets the random seed for all random number generators and disables cuDNN's nondeterministic algorithms.

    :param seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train_loop(model: torch.nn.Module, train_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset,
               collate_fn: Callable, device: torch.device, batch_size: int, max_epochs: int, lr: float,
               patience: int, exponential_lr: bool = False, exponential_lr_gamma: float = 1.0,
               checkpoint_callback: Optional[Callable] = None, name: str = 'model',
               clear_output_each_epoch=True, best_model_temp_path="best_model.pt",
               optimize_metric: str = "em", minimize_metric: bool = False) -> \
        tuple[torch.nn.Module, list[float], list[float], list[float], list[float]]:
    """
    Training loop for a given model and dataset.

    :param model: The model to train. (PyTorch model)
    :param train_set: The training set. (PyTorch dataset)
    :param val_set: The validation set. (PyTorch dataset)
    :param collate_fn: The collate function to use for input batches.
    :param device: The device to use for training.
    :param batch_size: The batch size to use.
    :param max_epochs: The maximum number of epochs to train for.
    :param lr: The learning rate to use.
    :param patience: The number of epochs without improvement to wait before stopping training.
    :param exponential_lr: Whether to use exponential learning rate day.
    :param exponential_lr_gamma: The gamma parameter for exponential learning rate decay.
    :param checkpoint_callback: An optional callback function to use for storing the best model between epochs.
    :param name: The name of the model (used for reporting and saving the model).
    :param clear_output_each_epoch: Whether to clear the output each epoch or not (in Jupyter notebooks).
    :param best_model_temp_path: The path to save the best model settings to (deleted at the end of training).
    :param optimize_metric: The metric to optimize for ("loss", "acc" or "em")
    :param minimize_metric: Whether to minimize the metric or not.
    :return: The best model and the training and validation losses, accuracy and EM scores over time.
    """
    # Create a dataloader for train and validation sets
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)
    # We use AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    # If we want to use exponential learning rate decay, we use a scheduler
    scheduler = None
    if exponential_lr:
        scheduler = ExponentialLR(optimizer, gamma=exponential_lr_gamma)

    # We keep track of the best model and the best metric
    best_metric = float('inf') if minimize_metric else -float('inf')
    # We keep track of the number of epochs without improvement
    no_improve_epoch = 0
    # We keep track of the losses over time and metrics
    train_losses = []
    val_losses = []
    accs = []
    ems = []

    # Move the model to the device
    model.to(device)

    for epoch in range(max_epochs):
        # TRAINING
        model.train()
        epoch_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training {name}, Epoch {epoch + 1}/{max_epochs}", ascii=" ="):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # forward pass
            optimizer.zero_grad()
            output, loss = model(**batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # VALIDATION
        model.eval()
        epoch_eval_loss = 0
        inputs = []
        outputs = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validating {name}, Epoch {epoch + 1}/{max_epochs}", ascii=" ="):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                output, loss = model(**batch)
                epoch_eval_loss += loss.item()

                # Add the inputs, outputs and labels to the lists
                inputs.append(batch["input_ids"])
                outputs.append(output)
                labels.append(batch["labels"])

        # EPOCH STEP
        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_val_loss = epoch_eval_loss / len(val_dataloader)
        # Calculate accuracy and EM
        acc = calculate_accuracy(outputs, labels)
        em = calculate_em(inputs, outputs, labels)
        # If we use dynamic learning rate, do a scheduler step on the validation loss
        if exponential_lr:
            scheduler.step()
        # Save the metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        accs.append(acc)
        ems.append(em)
        # Check if we need to save the model or stop training
        prio_metric = acc if optimize_metric == "acc" else em if optimize_metric == "em" else avg_val_loss
        if prio_metric < best_metric if minimize_metric else prio_metric > best_metric:
            best_metric = prio_metric
            torch.save(model.state_dict(), best_model_temp_path)
            if checkpoint_callback is not None:
                checkpoint_callback(model)
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print("Early stopping.")
                break

        if clear_output_each_epoch:
            # Clear output to avoid cluttering
            clear_output(wait=True)

        # Print information about the epoch that just passed
        print(f"Epoch Report: Epoch {epoch + 1}/{max_epochs}, Model {name}\n"
              f"- Train loss: {avg_train_loss:.4f}\n"
              f"- Val loss: {avg_val_loss:.4f}\n"
              f"- Acc: {acc:.4f}\n"
              f"- EM: {em:.4f}")
        print(
            f"Optimizing for {'EM' if optimize_metric == 'em' else optimize_metric.capitalize()}, current best: {best_metric:.4f} "
            f"({('From this epoch' if no_improve_epoch == 0 else f'No improvement for {no_improve_epoch} epoch' + ('s' if no_improve_epoch > 1 else ''))})")

    # Load the best model
    model.load_state_dict(torch.load(best_model_temp_path))
    # Delete the temporary file
    os.remove(best_model_temp_path)
    return model, train_losses, val_losses, accs, ems


def eval_loop(model: torch.nn.Module, test_set: torch.utils.data.Dataset, collate_fn: Callable,
              device: torch.device, batch_size: int = 16) -> \
        Tuple[float, float, float]:
    """
    Evaluation loop for a given model and dataset.

    :param model: The model to evaluate. (PyTorch model)
    :param test_set: The test set. (PyTorch dataset)
    :param collate_fn: The collate function to use for the test set.
    :param device: The device to use for evaluation.
    :param batch_size: The batch size to use.
    embeddings.
    :return: The test loss, accuracy and the EM score of the model on the test dataset.
    """
    # Set up dataloader for test set
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Keep track of the total loss
    total_test_loss = 0

    # Move the model to the device
    model.to(device)

    # Test the model on the test set
    model.eval()
    inputs = []
    outputs = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", ascii=" ="):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # forward pass
            output, loss = model(**batch)
            total_test_loss += loss.item()
            # Store the inputs, outputs and labels for later
            inputs.append(batch["input_ids"])
            outputs.append(output)
            labels.append(batch["labels"])

    # Compute metrics
    avg_loss = total_test_loss / len(test_dataloader)
    acc = calculate_accuracy(outputs, labels)
    em = calculate_em(inputs, outputs, labels)

    print(f'Test Loss: {avg_loss}')
    print(f'Accuracy: {acc}')
    print(f'EM: {em}')

    return avg_loss, acc, em


def store_model(model, path) -> None:
    """
    Stores a model checkpoint to the given filepath

    :param model: The model to store.
    :param path: The path to store the model to.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path) -> torch.nn.Module:
    """
    Loads a model checkpoint from the given filepath

    :param model: An instance of the model to load the checkpoint into.
    :param path: The path to load the model from.
    :return: The model with the loaded checkpoint.
    """
    model.load_state_dict(torch.load(path))
    return model


def load_eval(eval_path: str, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param eval_path: The path to the evaluation results.
    :param models: A list of models to load the evaluation results for.
    :return: A tuple of two dataframes, the first one containing the best model information and the second one
    containing the epoch information.
    """
    best_model_info = []
    epoch_info = []

    for model in models:
        path = eval_path.format(model)
        with open(path, 'rb') as f:
            results = pickle.load(f)

        for classifier in results:
            name, train_losses, val_losses, accs, ems, test_loss, acc, em = classifier
            best_epoch = ems.index(max(ems)) + 1
            best_model_info.append((model, best_epoch, test_loss, acc, em))

            for epoch in range(len(train_losses)):
                epoch_info.append((name, epoch + 1, train_losses[epoch], val_losses[epoch], accs[epoch], ems[epoch]))

    eval_df = pd.DataFrame(best_model_info, columns=["model", "best_epoch", "test_loss", "acc", "em"])
    eval_df[["acc", "em"]] = eval_df[["acc", "em"]].round(3)

    epoch_df = pd.DataFrame(epoch_info, columns=["model", "epoch", "train_loss", "val_loss", "acc", "em"])
    epoch_df[["train_loss", "val_loss", "acc", "em"]] = epoch_df[["train_loss", "val_loss", "acc", "em"]].round(3)

    return eval_df, epoch_df
