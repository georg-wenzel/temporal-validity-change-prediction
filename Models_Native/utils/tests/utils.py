import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        """
        This is a dummy model that returns a fixed set of 12 predictions, meant to be used with the data/TestDataset.csv
        """

        super(DummyModel, self).__init__()
        self.predictions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  # 3 correct, 1EM
                                         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # 2 correct, 0EM
                                         [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # 0 correct, 0EM
                                         [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # 1 correct, 0EM
        self.start_index = 0
        self.loss = torch.nn.CrossEntropyLoss()
        self.linear = torch.nn.Linear(128, 3)

    def forward(self, input_ids, labels=None, **kwargs):
        """
        Returns a fixed set of predictions, and a loss if labels are provided.

        :param input_ids: Input IDs (transformer input)
        :param labels: Labels for the input IDs
        :param kwargs: Any other arguments are ignored in this model.
        :return: A tuple of (predictions, loss) if labels are provided, otherwise (predictions, None)
        """
        bs = input_ids.shape[0]
        preds = self.predictions[self.start_index:self.start_index + bs]
        self.start_index = (self.start_index + bs) % self.predictions.shape[0]
        dummy_in = torch.randn(bs, 128)
        dummy_out = self.linear(dummy_in)
        if labels is not None:
            loss = self.loss(dummy_out, labels)
            return preds, loss
        else:
            return preds, None

    def __str__(self):
        return "DummyModel"

    def __repr__(self):
        return "DummyModel"
