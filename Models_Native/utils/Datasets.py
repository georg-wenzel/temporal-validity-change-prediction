import pandas as pd
import torch
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    def __init__(self, csv_file, tokenizer, split_input=False, output_columns=None, scale_columns=None):
        """
        Dataset class for the Twitter dataset. If output_columns are not provided, the default is to use the "change"
        column as the label. If scale_columns are not provided, the default is to not scale any columns. Scaling can
        only be used for output columns and applies linear min-max scaling to the specified columns.

        :param csv_file: Path to the csv file containing the dataset.
        :param tokenizer: Tokenizer to use for encoding the inputs.
        :param split_input: Whether to split the input into two parts (context and follow-up).
        :param output_columns: Dictionary mapping column names in the csv file to names of output columns.
        :param scale_columns: Dictionary mapping names of output columns to a tuple of (min, max) values before scaling.
        """
        if output_columns is None:
            output_columns = {'change': 'labels'}
        if scale_columns is None:
            scale_columns = {}

        # Read the dataset from the csv file
        self.data = pd.read_csv(csv_file).reset_index(drop=True)
        # Store the column mapping
        self.col_mapping = output_columns
        # Store the split input flag
        self.split_input = split_input
        # Extract both parts of the input
        self.target = self.data['context'].values
        self.followup = self.data['follow_up'].values
        # Extract all required output columns
        self.labels = {v: self.data[k].values for k, v in output_columns.items()}
        # Set the tokenizer
        self.tokenizer = tokenizer

        # The "change" labels are strings, so if they are part of the output, we encode them using the label_map
        self.label_map = {
            'decreased': 0,
            'neutral': 1,
            'increased': 2
        }

        # Apply the label map to the change labels if they are part of the output
        if 'change' in output_columns:
            target_col = output_columns['change']
            self.labels[target_col] = [self.label_map[label] for label in self.labels[target_col]]

        # Scale the specified output columns to the range [0, 1]
        for col in scale_columns:
            def scale_func(x):
                return (x - scale_columns[col][0]) / (scale_columns[col][1] - scale_columns[col][0])

            self.labels[col] = list(map(scale_func, self.labels[col]))

    def __len__(self):
        """
        Returns the length of the dataset.
        :return: The length of the dataset (number of samples).
        """
        return len(self.data)

    def get_labels(self):
        """
        Returns the labels of the dataset as a pytorch tensor. Defaults to the "change" labels if they are part of the
        output, otherwise returns the first output column. This is used for stratified splitting into train/val set.
        :return: The labels of the dataset as a pytorch tensor.
        """
        # by default, we return the change labels, if they are not part of the output we return the first output column
        if 'change' in self.col_mapping:
            return torch.tensor(self.labels[self.col_mapping['change']])
        else:
            return torch.tensor(self.labels[list(self.labels.keys())[0]])

    def get_target_statements(self):
        """
        Returns the target statements of the dataset as a list.
        :return: The target statements of the dataset as a list.
        """
        return self.target

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset. If split_input is True, the input is split into two parts (target
        and follow-up) and each part is tokenized separately. Otherwise, the input is tokenized as a single string.

        :param idx: The index of the sample to return.
        :return: A dictionary containing the input_ids, attention_mask, and token_type_ids (if applicable) for the
        sample, as well as the labels for the sample.
        """
        out_target = self.target[idx]
        out_followup = self.followup[idx]
        out = {k: torch.tensor(self.labels[k][idx]) for k in self.labels.keys()}

        # If we want to split the input into two parts, we tokenize both the target and follow-up sentence
        # and return a pytorch tensor for each.
        if self.split_input:
            out_target = self.tokenizer.encode_plus(
                out_target,
                return_tensors='pt'
            )
            out_followup = self.tokenizer.encode_plus(
                out_followup,
                return_tensors='pt'
            )
            out['input_ids'] = out_target['input_ids'].squeeze()
            out['attention_mask'] = out_target['attention_mask'].squeeze()
            if 'token_type_ids' in out_target:
                out['token_type_ids'] = out_target['token_type_ids'].squeeze()
            out['input_ids_followup'] = out_followup['input_ids'].squeeze()
            out['attention_mask_followup'] = out_followup['attention_mask'].squeeze()
            if 'token_type_ids' in out_followup:
                out['token_type_ids_followup'] = out_followup['token_type_ids'].squeeze()
        # If we don't want to split the input, we tokenize the combined target and follow-up sentence
        # and return a single pytorch tensor.
        else:
            out_concat = self.tokenizer.encode_plus(
                out_target, out_followup,
                return_tensors='pt'
            )
            out['input_ids'] = out_concat['input_ids'].squeeze()
            out['attention_mask'] = out_concat['attention_mask'].squeeze()
            if 'token_type_ids' in out_concat:
                out['token_type_ids'] = out_concat['token_type_ids'].squeeze()

        return out

    def get_label_map(self):
        """
        Returns the reversed label map for the dataset.
        :return: The label map for the dataset, mapping from label indices to label names.
        """
        return {v: k for k, v in self.label_map.items()}
