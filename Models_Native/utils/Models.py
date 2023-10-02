import torch
from torch import nn
import torch.nn.functional as F


class SentenceEmbeddingModule(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True):
        """
        Submodule that takes in a transformer-based model and returns the embedding of the [CLS] token via the
        pooler_output attribute of the model. The output size of the module is equal to the hidden size
        of the transformer model.

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        """
        super(SentenceEmbeddingModule, self).__init__()
        self.embedding_model = embedding_model
        self.out_size = self.embedding_model.config.hidden_size
        self.max_tokens = self.embedding_model.config.max_position_embeddings
        if freeze_embedding_model:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
        else:
            for param in self.embedding_model.parameters():
                param.requires_grad = True
        # Unfreeze the pooler layer
        for param in self.embedding_model.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Forward pass of the module. Returns the embedding of the [CLS] token.

        :param input_ids: The input ids of the input sequence.
        :param attention_mask: The attention mask of the input sequence.
        :param token_type_ids: The token type ids of the input sequence (optional).
        :param kwargs: Additional keyword arguments (unused).
        :return: The sentence embedding.
        """
        if token_type_ids is None:
            output = self.embedding_model(input_ids, attention_mask=attention_mask)
        else:
            output = self.embedding_model(input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)
        return output.pooler_output


class TokenEmbeddingModule(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True):
        """
        Submodule that takes in a transformer-based model and returns the token embeddings.
        The output size of the module is equal to (max_tokens, hidden_size))

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        """
        super(TokenEmbeddingModule, self).__init__()
        self.embedding_model = embedding_model
        self.out_size = self.embedding_model.config.hidden_size
        self.max_tokens = self.embedding_model.config.max_position_embeddings
        if freeze_embedding_model:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
        else:
            for param in self.embedding_model.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Forward pass of the module. Returns the token embeddings

        :param input_ids: The input ids of the input sequence.
        :param attention_mask: The attention mask of the input sequence.
        :param token_type_ids: The token type ids of the input sequence (optional).
        :param kwargs: Additional keyword arguments (unused).
        :return: The token embeddings.
        """
        if token_type_ids is None:
            output = self.embedding_model(input_ids, attention_mask=attention_mask)
        else:
            output = self.embedding_model(input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)
        return output.last_hidden_state


# Submodule that provides a shared hidden layer for both sentences after separately embedding them
class SiameseEmbeddingModule(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True, hidden_size=768,
                 detailed_vector_concatenation=True):
        """
        Submodule that provides a shared hidden layer for both sentences after separately embedding them.
        The output size of the module is equal to the specified hidden size.

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        :param hidden_size: The size of the output hidden layer.
        :param detailed_vector_concatenation: If set to true, the linear layer in this module will be based on
        the representation (h1, h2, h1-h2, h1*h2). Otherwise, it will be based on (h1, h2).
        """
        super(SiameseEmbeddingModule, self).__init__()
        self.embedding_module = SentenceEmbeddingModule(embedding_model, freeze_embedding_model)
        self.detailed = detailed_vector_concatenation
        if self.detailed:
            # Use h1, h2, h1-h2, h1*h2
            self.linear = nn.Linear(self.embedding_module.out_size * 4, hidden_size)
        else:
            # Use h1, h2
            self.linear = nn.Linear(self.embedding_module.out_size * 2, hidden_size)
        self.relu = nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')

    def forward(self, input_ids, attention_mask, input_ids_followup, attention_mask_followup,
                token_type_ids=None, token_type_ids_followup=None, **kwargs):
        """
        Forward pass of the module. Returns an intermediate representation of dimensionality hidden_size.
        The input sequences are embedded separately and then concatenated. ReLU is applied to the output of the layer.

        :param input_ids: The input ids of the target sequence.
        :param attention_mask: The attention mask of the target sequence.
        :param input_ids_followup: The input ids of the followup sequence.
        :param attention_mask_followup: The attention mask of the followup sequence.
        :param token_type_ids: The token type ids of the target sequence (optional).
        :param token_type_ids_followup: The token type ids of the followup sequence (optional).
        :param kwargs: Additional keyword arguments (unused).
        :return: The intermediate representation of size hidden_size.
        """
        # Get the embeddings
        if token_type_ids is None:
            embedding_target = self.embedding_module(input_ids, attention_mask)
            embedding_followup = self.embedding_module(input_ids_followup, attention_mask_followup)
        else:
            embedding_target = self.embedding_module(input_ids, attention_mask,
                                                     token_type_ids=token_type_ids)
            embedding_followup = self.embedding_module(input_ids_followup, attention_mask_followup,
                                                       token_type_ids=token_type_ids_followup)

        # Concatenate the embeddings
        if self.detailed:
            combined = torch.cat([embedding_target, embedding_followup,
                                  embedding_target - embedding_followup,
                                  embedding_target * embedding_followup], dim=1)
        else:
            combined = torch.cat([embedding_target, embedding_followup], dim=1)

        # Pass the concatenated embeddings through the linear layer and apply ReLU
        output = self.linear(combined)
        output = self.relu(output)
        return output


class TransformerEmbeddingModule(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True, hidden_size=768):
        """
        Submodule that provides a shared hidden layer for both sentences after embedding them together.
        The output size of the module is equal to the specified hidden size.

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        :param hidden_size: The size of the output hidden layer.
        """
        super(TransformerEmbeddingModule, self).__init__()
        self.embedding_module = SentenceEmbeddingModule(embedding_model, freeze_embedding_model)
        self.linear = nn.Linear(self.embedding_module.out_size, hidden_size)
        self.relu = nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        """
        Forward pass of the module. Returns an intermediate representation of dimensionality hidden_size.
        The input sequence is embedded and then passed through a linear layer. ReLU is applied to the output.

        :param input_ids: The input ids of the sequence.
        :param attention_mask: The attention mask of the sequence.
        :param token_type_ids: The token type ids of the sequence (optional).
        :param kwargs: Additional keyword arguments (unused).
        :return: The intermediate representation of size hidden_size.
        """
        # Get the embedding
        embedding = self.embedding_module(input_ids, attention_mask, token_type_ids)
        # Pass the embedding through a linear layer and apply ReLU
        output = self.linear(embedding)
        output = self.relu(output)
        return output


class SiameseClassifier(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True, hidden_size=768,
                 detailed_vector_concatenation=True, dropout=0.1, num_classes=3):
        """
        Siamese model with transformer-based embeddings. The hidden layer representation of the two sentences is
        generated by a SiameseEmbeddingModule. To the output of this model, dropout is applied, and then a linear
        layer is used to classify the two sentences.

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        :param hidden_size: The size of the hidden layer.
        :param detailed_vector_concatenation: Whether to use the detailed vector concatenation method.
        :param dropout: The dropout rate to apply to the output of the embedding module.
        :param num_classes: The number of classes to classify the two sentences into (output logits are 0-num_classes)
        """
        super(SiameseClassifier, self).__init__()
        # Store dropout
        self.dropout_rate = dropout
        # We use CrossEntropyLoss as our loss function
        self.loss = nn.CrossEntropyLoss()
        # Model structure
        self.concatenation_module = SiameseEmbeddingModule(embedding_model, freeze_embedding_model, hidden_size,
                                                           detailed_vector_concatenation)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, input_ids_followup, attention_mask_followup,
                token_type_ids=None, token_type_ids_followup=None, labels=None):
        """
        Forward pass of the model. Returns the logits and the loss if labels are given.

        :param input_ids: The input ids of the target sequence.
        :param attention_mask: The attention mask of the target sequence.
        :param input_ids_followup: The input ids of the followup sequence.
        :param attention_mask_followup: The attention mask of the followup sequence.
        :param token_type_ids: The token type ids of the target sequence (optional).
        :param token_type_ids_followup: The token type ids of the followup sequence (optional).
        :param labels: The target labels (optional).
        :return: The logits, and the loss if labels are given.
        """
        # On forward pass, we pass the inputs through the concatenation module to get an embedding of hidden_size
        output = self.concatenation_module(input_ids, attention_mask, input_ids_followup,
                                           attention_mask_followup, token_type_ids, token_type_ids_followup)
        # We apply dropout if specified
        if self.dropout_rate > 0:
            output = self.dropout(output)
        # We pass the output through a linear layer to get the logits
        output = self.classifier(output)

        # If we are given labels, we calculate the loss and return it along with the output
        loss = None
        if labels is not None:
            loss = self.loss(output, labels)
        # We return the outputs using softmax to get probabilities
        return F.softmax(output, dim=1), loss


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True, hidden_size=768, dropout=0.1, num_classes=3):
        super(TransformerClassifier, self).__init__()
        """
        Transformer-based classifier. The hidden layer representation of the two sentences is
        generated by a TransformerEmbeddingModule. To the output of this model, dropout is applied, and then a linear
        layer is used to classify the embedding.
        
        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        :param hidden_size: The size of the hidden layer.
        :param dropout: The dropout rate to apply to the output of the embedding module.
        :param num_classes: The number of classes to classify the two sentences into (output logits are 0-num_classes)
        """
        # Store dropout
        self.dropout_rate = dropout
        # We use CrossEntropyLoss as our loss function
        self.loss = nn.CrossEntropyLoss()
        # Model structure
        self.concatenation_module = TransformerEmbeddingModule(embedding_model, freeze_embedding_model, hidden_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass of the module. Returns the output of the classifier, and the loss if labels are given.

        :param input_ids: The input ids of the sequence.
        :param attention_mask: The attention mask of the sequence.
        :param token_type_ids: The token type ids of the sequence (optional).
        :param labels: The target labels (optional).
        :return: The logits, and the loss if labels are given.
        """
        # On forward pass, we get embeddings for both sentences from the same embedding pass
        output = self.concatenation_module(input_ids, attention_mask, token_type_ids)
        # We apply dropout if specified
        if self.dropout_rate > 0:
            output = self.dropout(output)
        # We pass the output through a linear layer to get the logits
        output = self.classifier(output)

        # If we are given labels, we calculate the loss and return it along with the output
        loss = None
        if labels is not None:
            loss = self.loss(output, labels)
        # We return the outputs using softmax to get probabilities
        return F.softmax(output, dim=1), loss


# Classifier with multitask learning on the duration and change labels
class MultiTaskClassifier(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=True, hidden_size=768,
                 detailed_vector_concatenation=True, dropout=0.1, cls_type="siamese", num_classes=3,
                 multitask_multiplier=1.0):
        """
        Multi-task classifier. The hidden layer representation of the two sentences is generated by a
        SiameseEmbeddingModule or a TransformerEmbeddingModule. To the output of this model, dropout is applied, and
        the hidden representation is used to classify three different tasks:
        - Predicting the TV duration of the target sentence (regression task)
        - Predicting the TV duration of the target sentence under the context of the followup sentence (regression task)
        - Predicting the change in TV duration of the target sentence under the context of the followup sentence
        (classification task)
        MSE is used as the loss function for the regression tasks, and CrossEntropyLoss is used for the classification
        task. Only the logits of the classification task are returned.

        :param embedding_model: The transformer model used for embedding the input.
        :param freeze_embedding_model: Whether to train parameters of the embedding model. In any case, the pooler
        layer remains unfrozen.
        :param hidden_size: The size of the hidden layer.
        :param detailed_vector_concatenation: Whether to use the detailed vector concatenation method in the Siamese
        embedding module. This is ignored if the transformer embedding module is used.
        :param dropout: The dropout rate to apply to the output of the embedding module.
        :param cls_type: The type of classifier to use. Can be "siamese" or "transformer".
        :param num_classes: The number of classes to classify the two sentences into (output logits are 0-num_classes)
        :param multitask_multiplier: The multiplier to apply to the loss of the multitask learning tasks.
        """
        super(MultiTaskClassifier, self).__init__()
        # Store dropout
        self.dropout_rate = dropout
        # We use CrossEntropyLoss as our loss function for the classification task
        self.loss_ce = nn.CrossEntropyLoss()
        # We use MSELoss as our loss function for the regression tasks
        self.loss_mse = nn.MSELoss()
        # Store the multiplier for the multitask learning tasks
        self.multitask_multiplier = multitask_multiplier

        # Model structure
        if cls_type == "siamese":
            self.concatenation_module = SiameseEmbeddingModule(embedding_model, freeze_embedding_model, hidden_size,
                                                               detailed_vector_concatenation)
        elif cls_type == "transformer":
            self.concatenation_module = TransformerEmbeddingModule(embedding_model, freeze_embedding_model,
                                                                   hidden_size)
        else:
            raise ValueError(f"Invalid classifier type: {cls_type}")

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)

        # Classifier layers for the different tasks
        self.cls_duration_standalone = nn.Linear(hidden_size, 1)
        self.cls_duration_shared = nn.Linear(hidden_size, 1)
        self.cls_change = nn.Linear(hidden_size, num_classes)

    def forward(self, labels_duration_standalone=None, labels_duration_combined=None,
                labels=None, **kwargs):
        """
        Forward pass of the module. Returns the output of the classifier, and the loss if labels are given.

        :param labels_duration_standalone: The target labels for the duration of the target sentence.
        :param labels_duration_combined: The target labels for the duration of the target sentence under the context
        of the followup sentence.
        :param labels: The target labels for the change in duration of the target sentence under the context
        of the followup sentence.
        :param kwargs: The input arguments for the embedding module. Should contain input_ids, attention_mask, and
        token_type_ids (optional), split into two sequences for the Siamese model.
        :return: The logits, and the loss if labels are given. Only the logits of the classification task are returned.
        """
        # Forward pass to get the hidden embedding
        output = self.concatenation_module(**kwargs)
        # We apply dropout if specified
        if self.dropout_rate > 0:
            output = self.dropout(output)
        # We pass the hidden embedding through the different classifiers
        output_duration_standalone = torch.sigmoid(self.cls_duration_standalone(output)).squeeze()
        output_duration_shared = torch.sigmoid(self.cls_duration_shared(output)).squeeze()
        output_change = self.cls_change(output)

        # If we are given labels, we calculate the loss and return it along with the output
        loss = None
        if labels_duration_standalone is not None:
            # Sum up the losses over the different tasks
            loss = self.multitask_multiplier * \
                   (self.loss_mse(output_duration_standalone, labels_duration_standalone.float()) +
                    self.loss_mse(output_duration_shared, labels_duration_combined.float())) + \
                   self.loss_ce(output_change, labels)
        # We return the outputs using softmax to get probabilities
        return F.softmax(output_change, dim=1), loss
