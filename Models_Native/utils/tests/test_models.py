import unittest

import torch
from transformers import BertModel

from Models_Native.utils.Models import SentenceEmbeddingModule, SiameseEmbeddingModule, TransformerEmbeddingModule, \
    SiameseClassifier, TransformerClassifier, MultiTaskClassifier, TokenEmbeddingModule
from Models_Native.utils.SelfExplainModels import InterpretationModel, SICModel, ExplainableModel


class TestModels(unittest.TestCase):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    pooler_parameters = set(id(p) for p in bert_model.pooler.parameters())

    def test_embedder(self):
        # Test unfrozen model
        embedder = SentenceEmbeddingModule(self.bert_model, freeze_embedding_model=False)
        # Test that the embedding model is not frozen
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")

        # Test frozen model
        embedder = SentenceEmbeddingModule(self.bert_model, freeze_embedding_model=True)
        # Test that the embedding model is frozen (except for the pooler layer)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs = embedder(input_ids, attention_mask)
        self.assertEqual((4, self.bert_model.config.hidden_size), outputs.shape, "Output shape is incorrect.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens = embedder(input_ids, attention_mask, token_type_ids)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

    def test_token_embedder(self):
        # Test unfrozen model
        embedder = TokenEmbeddingModule(self.bert_model, freeze_embedding_model=False)
        # Test that the embedding model is not frozen
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")

        # Test frozen model
        embedder = TokenEmbeddingModule(self.bert_model, freeze_embedding_model=True)
        # Test that the embedding model is frozen
        for param in self.bert_model.parameters():
            self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 200, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs = embedder(input_ids, attention_mask)
        self.assertEqual((4, embedder.max_tokens, self.bert_model.config.hidden_size), outputs.shape,
                         "Output shape is incorrect.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens = embedder(input_ids, attention_mask, token_type_ids)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

    def test_siamese_embedding(self):
        # Test if unfrozen parameters get passed down properly
        embedder = SiameseEmbeddingModule(self.bert_model, freeze_embedding_model=False)
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")

        # Test if frozen parameters get passed down properly
        embedder = SiameseEmbeddingModule(self.bert_model, freeze_embedding_model=True, hidden_size=200)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        input_ids_followup = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs = embedder(input_ids, attention_mask, input_ids_followup, attention_mask_followup)
        self.assertEqual((4, 200), outputs.shape, "Output shape is incorrect (should be hidden layer size).")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        token_type_ids_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens = embedder(input_ids, attention_mask, input_ids_followup, attention_mask_followup,
                                      token_type_ids, token_type_ids_followup)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

        # Test that the output is different when detailed vector concatenation is used
        embedder = SiameseEmbeddingModule(self.bert_model, freeze_embedding_model=True, hidden_size=200,
                                          detailed_vector_concatenation=True)
        outputs_detailed = embedder(input_ids, attention_mask, input_ids_followup, attention_mask_followup)
        self.assertFalse(torch.equal(outputs, outputs_detailed),
                         "Outputs should be different when detailed vector concatenation is used.")

    def test_transformer_embedding(self):
        # Test if unfrozen parameters get passed down properly
        embedder = TransformerEmbeddingModule(self.bert_model, freeze_embedding_model=False)
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")

        # Test if frozen parameters get passed down properly
        embedder = TransformerEmbeddingModule(self.bert_model, freeze_embedding_model=True, hidden_size=200)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs = embedder(input_ids, attention_mask)
        self.assertEqual((4, 200), outputs.shape,
                         "Output shape is incorrect (should be hidden layer size).")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens = embedder(input_ids, attention_mask, token_type_ids)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

    def test_siamese_classifier(self):
        classifier = SiameseClassifier(self.bert_model, freeze_embedding_model=False)

        # Test if unfrozen parameters get passed down properly
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")
        for param in classifier.parameters():
            self.assertTrue(param.requires_grad, "Classifier is frozen, but should not be.")

        # Test if frozen parameters get passed down properly
        classifier = SiameseClassifier(self.bert_model, freeze_embedding_model=True, hidden_size=200,
                                       detailed_vector_concatenation=True, dropout=0.3, num_classes=5)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test if the remainder of the parameters get passed down to the embedder
        self.assertEqual(200, classifier.concatenation_module.linear.out_features,
                         "Linear layer input size is incorrect.")
        self.assertEqual(True, classifier.concatenation_module.detailed,
                         "Detailed vector concatenation should be True, but is not.")
        # Test if an appropriate dropout layer is defined
        self.assertEqual(0.3, classifier.dropout.p, "Dropout probability is incorrect or layer does not exist.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        input_ids_followup = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs, loss = classifier(input_ids, attention_mask, input_ids_followup, attention_mask_followup)
        self.assertEqual((4, 5), outputs.shape, "Output shape is incorrect.")
        self.assertIsNone(loss, "Loss should be None when no labels are provided.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        token_type_ids_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens, _ = classifier(input_ids, attention_mask, input_ids_followup, attention_mask_followup,
                                           token_type_ids, token_type_ids_followup)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

        # Test that loss is provided if labels are provided
        labels = torch.randint(0, 5, (4,))
        outputs, loss = classifier(input_ids, attention_mask, input_ids_followup, attention_mask_followup,
                                   labels=labels)
        self.assertIsNotNone(loss, "Loss should be provided when labels are provided.")

    def test_transformer_classifier(self):
        classifier = TransformerClassifier(self.bert_model, freeze_embedding_model=False)
        # Test if unfrozen parameters get passed down properly
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")
        for param in classifier.parameters():
            self.assertTrue(param.requires_grad, "Classifier is frozen, but should not be.")

        # Test if frozen parameters get passed down properly
        classifier = TransformerClassifier(self.bert_model, freeze_embedding_model=True, hidden_size=200,
                                           dropout=0.3, num_classes=5)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test if the remainder of the parameters get passed down to the embedder
        self.assertEqual(200, classifier.concatenation_module.linear.out_features,
                         "Linear layer input size is incorrect.")
        # Test if an appropriate dropout layer is defined
        self.assertEqual(0.3, classifier.dropout.p, "Dropout probability is incorrect or layer does not exist.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs, loss = classifier(input_ids, attention_mask)
        self.assertEqual((4, 5), outputs.shape, "Output shape is incorrect.")
        self.assertIsNone(loss, "Loss should be None when no labels are provided.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens, _ = classifier(input_ids, attention_mask, token_type_ids)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

        # Test that loss is provided if labels are provided
        labels = torch.randint(0, 5, (4,))
        outputs, loss = classifier(input_ids, attention_mask,
                                   labels=labels)
        self.assertIsNotNone(loss, "Loss should be provided when labels are provided.")

    def test_multitask_siamese_classifier(self):
        classifier = MultiTaskClassifier(self.bert_model, freeze_embedding_model=False)
        # Test if unfrozen parameters get passed down properly
        for param in self.bert_model.parameters():
            self.assertTrue(param.requires_grad, "Embedding model is frozen, but should not be.")
        for param in classifier.parameters():
            self.assertTrue(param.requires_grad, "Classifier is frozen, but should not be.")

        # Test if frozen parameters get passed down properly
        classifier = MultiTaskClassifier(self.bert_model, freeze_embedding_model=True, hidden_size=200,
                                         dropout=0.3, detailed_vector_concatenation=True,
                                         cls_type="siamese", num_classes=5)
        for param in self.bert_model.parameters():
            if id(param) in self.pooler_parameters:
                self.assertTrue(param.requires_grad, "Pooler layer is frozen, but should not be.")
            else:
                self.assertFalse(param.requires_grad, "Embedding model is not frozen, but should be.")

        # Test if the remainder of the parameters get passed down to the embedder
        self.assertEqual(200, classifier.concatenation_module.linear.out_features,
                         "Linear layer input size is incorrect.")
        self.assertEqual(True, classifier.concatenation_module.detailed,
                         "Detailed vector concatenation should be True, but is not.")
        # Test if an appropriate dropout layer is defined
        self.assertEqual(0.3, classifier.dropout.p, "Dropout probability is incorrect or layer does not exist.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        input_ids_followup = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs, loss = classifier(input_ids=input_ids, attention_mask=attention_mask,
                                   input_ids_followup=input_ids_followup,
                                   attention_mask_followup=attention_mask_followup)
        self.assertEqual((4, 5), outputs.shape, "Output shape is incorrect.")
        self.assertIsNone(loss, "Loss should be None when no labels are provided.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        token_type_ids_followup = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens, _ = classifier(input_ids=input_ids, attention_mask=attention_mask,
                                           input_ids_followup=input_ids_followup,
                                           attention_mask_followup=attention_mask_followup,
                                           token_type_ids=token_type_ids,
                                           token_type_ids_followup=token_type_ids_followup)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

        # Test that loss is provided if labels are provided
        labels_duration_standalone = torch.randint(0, 5, (4,))
        labels_duration_combined = torch.randint(0, 5, (4,))
        labels_change = torch.randint(0, 5, (4,))
        outputs, loss = classifier(input_ids=input_ids, attention_mask=attention_mask,
                                   input_ids_followup=input_ids_followup,
                                   attention_mask_followup=attention_mask_followup,
                                   labels_duration_standalone=labels_duration_standalone,
                                   labels_duration_combined=labels_duration_combined,
                                   labels=labels_change)
        self.assertIsNotNone(loss, "Loss should be provided when labels are provided.")

    def test_multitask_transformer_classifier(self):
        classifier = MultiTaskClassifier(self.bert_model, freeze_embedding_model=True, hidden_size=200,
                                         dropout=0.3, cls_type="transformer", num_classes=5)

        # Test if the remainder of the parameters get passed down to the embedder
        self.assertEqual(200, classifier.concatenation_module.linear.out_features,
                         "Linear layer input size is incorrect.")
        # Test if an appropriate dropout layer is defined
        self.assertEqual(0.3, classifier.dropout.p, "Dropout probability is incorrect or layer does not exist.")

        # Test the shape of inputs and outputs
        input_ids = torch.randint(0, 1000, (4, self.bert_model.config.max_position_embeddings))
        attention_mask = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs, loss = classifier(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual((4, 5), outputs.shape, "Output shape is incorrect.")
        self.assertIsNone(loss, "Loss should be None when no labels are provided.")

        # Test that token_type_ids are accepted (change the output values)
        token_type_ids = torch.randint(0, 2, (4, self.bert_model.config.max_position_embeddings))
        outputs_withtokens, _ = classifier(input_ids=input_ids, attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)
        self.assertFalse(torch.equal(outputs, outputs_withtokens),
                         "Outputs should be different when token_type_ids are used.")

        # Test that loss is provided if labels are provided
        labels_duration_standalone = torch.randint(0, 5, (4,))
        labels_duration_combined = torch.randint(0, 5, (4,))
        labels_change = torch.randint(0, 5, (4,))
        outputs, loss = classifier(input_ids=input_ids, attention_mask=attention_mask,
                                   labels_duration_standalone=labels_duration_standalone,
                                   labels_duration_combined=labels_duration_combined,
                                   labels=labels_change)
        self.assertIsNotNone(loss, "Loss should be provided when labels are provided.")

    def test_interpretation_model(self):
        model = InterpretationModel(100)
        # get random input data (bs, num_spans, hidden)
        input_data = torch.rand((4, 10, 100))
        # get span masks (bs, num_spans), set all to legal
        span_masks = torch.zeros((4, 10))
        # pass through model
        H, a_ij = model(input_data, span_masks)
        # check output shape
        # H = (bs, hidden)
        # a_ij = (bs, num_spans)
        self.assertEqual(H.shape, (4, 100))
        self.assertEqual(a_ij.shape, (4, 10))

    def test_SIC_model(self):
        model = SICModel(100)
        # get random input data (bs, length, hidden)
        input_data = torch.rand((4, 768, 100))
        # get indices
        start_indices = []
        end_indices = []
        for i in range(768):
            for j in range(i, 768):
                start_indices.append(i)
                end_indices.append(j)
        start_indices = torch.tensor(start_indices)
        end_indices = torch.tensor(end_indices)
        # pass through model
        h_ij = model(input_data, start_indices, end_indices)
        # check output shape
        # h_ij = (bs, num_spans, hidden)
        self.assertEqual(h_ij.shape, (4, len(start_indices), 100))

    def test_explainable_model(self):
        classifier = ExplainableModel(self.bert_model, freeze_embedding_model=True)
        # get max_tokens of bert model
        max_tokens = classifier.embedding.max_tokens
        # get random input data (bs, length)
        input_ids = torch.randint(0, 1000, (4, max_tokens))
        # get random attention mask (bs, length)
        attention_mask = torch.randint(0, 2, (4, max_tokens))
        # Create start and end indices
        start_indices = []
        end_indices = []
        for i in range(max_tokens):
            for j in range(i, max_tokens):
                start_indices.append(i)
                end_indices.append(j)
        start_indices = torch.tensor(start_indices)
        end_indices = torch.tensor(end_indices)
        # create span masks
        span_masks = torch.zeros((4, len(start_indices)))
        # pass through model
        outputs, loss = classifier(input_ids, attention_mask, start_indices, end_indices, span_masks)
        # check output shape
        # outputs = (bs, num_classes)
        self.assertEqual(outputs.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()
