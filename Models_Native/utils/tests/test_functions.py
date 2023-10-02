import os
import unittest
from functools import partial

import torch
from torch import device
from transformers import BertTokenizer

from Models_Native.utils.Datasets import TwitterDataset
from Models_Native.utils.Functions import collate_batch, train_loop, eval_loop, store_model, load_model, \
    self_explain_collate_batch, calculate_accuracy, calculate_em
from Models_Native.utils.tests.utils import DummyModel


class TestFunctions(unittest.TestCase):
    dataset_path = "./data/TestDataset.csv"

    def test_collate_batch(self):
        batch = [
            {'input_ids': torch.tensor([1, 2]), 'labels_change': torch.tensor(1)},
            {'input_ids': torch.tensor([4, 5, 6]), 'labels_change': torch.tensor(2)},
            {'input_ids': torch.tensor([7, 8, 9, 10]), 'labels_change': torch.tensor(0)}
        ]

        expected = {
            'input_ids': torch.tensor([[1, 2, 6, 6], [4, 5, 6, 6], [7, 8, 9, 10]]),
            'labels_change': torch.tensor([1, 2, 0])
        }

        actual = collate_batch(batch, 6)

        for dim in expected.keys():
            self.assertTrue(torch.equal(expected[dim], actual[dim]), f"The collated batch is incorrect. "
                                                                     f"(Dimension {dim})")

    def test_self_explain_collate_batch(self):
        batch = [
            {'input_ids': torch.tensor([3, 2]), 'labels_change': torch.tensor(1)},
            {'input_ids': torch.tensor([4, 5, 2]), 'labels_change': torch.tensor(2)},
            {'input_ids': torch.tensor([6, 7, 8, 2]), 'labels_change': torch.tensor(0)}
        ]

        # build the start and end indices
        start_indices = []
        end_indices = []
        for i in range(1, 3):
            for j in range(i, 3):
                start_indices.append(i)
                end_indices.append(j)

        # build the span masks
        span_masks = []
        for k in range(3):
            mask = []
            for i in range(1, 3):
                for j in range(i, 3):
                    token_idx = batch[k]["input_ids"].tolist().index(2)
                    token_len = len(batch[k]["input_ids"])
                    if j < token_idx or i > token_idx + 1 and j < token_len - 1:
                        mask.append(0)
                    else:
                        mask.append(1e6)
            span_masks.append(mask)

        expected = {
            'input_ids': torch.tensor([[3, 2, 1, 1], [4, 5, 2, 1], [6, 7, 8, 2]]),
            'labels_change': torch.tensor([1, 2, 0]),
            'start_indices': torch.LongTensor(start_indices),
            'end_indices': torch.LongTensor(end_indices),
            'span_masks': torch.LongTensor(span_masks)
        }
        actual = self_explain_collate_batch(batch, 1)

        for dim in expected.keys():
            self.assertTrue(torch.equal(expected[dim], actual[dim]), f"The collated batch is incorrect. "
                                                                     f"(Dimension {dim})")

    def test_calculate_accuracy(self):
        # outputs and labels are lists of batch results
        outputs = [
            torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]),
            torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.1, 0.7], [0.9, 0.1, 0.0]])
        ]
        labels = [
            torch.tensor([2, 1, 1]),
            torch.tensor([0, 1, 0]),
        ]
        # Here 3/6 are correct, so accuracy is 0.5 overall
        acc = calculate_accuracy(outputs, labels)
        self.assertEqual(0.5, acc, "Accuracy should be 50%.")

    def test_calculate_em(self):
        # inputs, outputs and labels are lists of batch results
        # inputs are the input_ids, in a real use case the sorting process should ensure that the same target
        # sequences are next to each other. In the test case, the target sequence is denoted by the first token
        # of each sequence. Each target sequence exists exactly three times.
        inputs = [
            torch.tensor([[5, 2, 9], [6, 2, 6], [7, 2, 8]]),
            torch.tensor([[6, 2, 2], [8, 2, 5], [6, 2, 9]]),
            torch.tensor([[5, 2, 3], [5, 2, 2], [8, 2, 4]]),
            torch.tensor([[8, 2, 0], [7, 2, 1], [7, 2, 7]]),
        ]

        outputs = [
            torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]),
            torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.1, 0.7], [0.9, 0.1, 0.0]]),
            torch.tensor([[0.5, 0.1, 0.4], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]),
            torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.6, 0.0], [0.1, 0.4, 0.5]])
        ]
        # correct labels and corresponding input sequences
        # [2, 1, 0] => inputs [1,2,3]
        # [1, 2, 0] => inputs [2,4,2]
        # [0, 2, 2] => inputs [1,1,4]
        # [2, 1, 2] => inputs [4,3,3]

        # we build labels so that all outputs corresponding to inputs 4 and 2 are correct
        # for 1 and 3, one output is incorrect.
        labels = [
            torch.tensor([1, 1, 0]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 2, 2]),
            torch.tensor([2, 0, 2])
        ]

        # the resulting accuracy should be 10/12 = 0.833
        acc = calculate_accuracy(outputs, labels)
        self.assertEqual(0.833, round(acc, 3), "Accuracy should be 83.3%.")

        # but the resulting EM should only be 2/4 = 0.5
        em = calculate_em(inputs, outputs, labels)
        self.assertEqual(0.5, em, "EM should be 50%.")

    def test_train_loop(self):
        # Create a dummy model
        model = DummyModel()
        # Store starting weights of linear layer
        start_weights = model.linear.weight.clone()
        # Load an arbitrary tokenizer to process the dataset
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load the test dataset as a TwitterDataset
        dataset_train = TwitterDataset(self.dataset_path, tokenizer)
        dataset_val = TwitterDataset(self.dataset_path, tokenizer)
        # Run the training loop
        model, train_losses, val_losses, accs, ems = train_loop(model, dataset_train, dataset_val,
                                                                partial(collate_batch, input_padding=0),
                                                                device('cpu'), max_epochs=1, batch_size=2, patience=1,
                                                                lr=0.001)
        # Check that we receive a model back
        self.assertIsInstance(model, DummyModel, "The train loop should return a model.")
        # Check that the weights of the linear layer have changed
        self.assertFalse(torch.equal(start_weights, model.linear.weight), "The weights of the linear layer should "
                                                                          "have changed during training.")

    def test_eval_function(self):
        # Create a dummy embedding model
        model = DummyModel()
        # Load an arbitrary tokenizer to process the dataset
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load the test dataset as a TwitterDataset
        dataset_test = TwitterDataset(self.dataset_path, tokenizer)
        # Run the evaluation loop
        test_loss, acc, em = eval_loop(model, dataset_test, partial(collate_batch, input_padding=0), device('cpu'),
                                       batch_size=2)
        # Check that acc and EM match what we expect
        self.assertEqual(acc, 0.5, "Accuracy for the dummy model and dataset should be 50%.")  # 6/12
        self.assertEqual(em, 0.25, "EM for the dummy model and dataset should be 25%.")  # 1/4

    def test_store_load_model(self):
        # Create a dummy model
        model = DummyModel()
        # Set the weights of the linear layer
        model.linear.weight.data.fill_(2.0)
        # Store the model
        model_path = "./data/test_model.pt"
        store_model(model, model_path)
        # Load the model
        model2 = DummyModel()
        model2 = load_model(model2, model_path)
        # Delete the stored file
        os.remove(model_path)
        # Check that the weights are the same
        self.assertTrue(torch.equal(model.linear.weight, model2.linear.weight), "The stored and loaded models "
                                                                                "should have the same weights.")


if __name__ == '__main__':
    unittest.main()
