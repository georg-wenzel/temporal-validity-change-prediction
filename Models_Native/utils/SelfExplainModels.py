# SelfExplain Model with alternative MultiTask implementation
# Compared to the SelfExplain model, the MultiTask model also predicts temporal validity durations as a byproduct.
# This code is based on https://github.com/ShannonAI/Self_Explaining_Structures_Improve_NLP_Models

import torch
from torch import nn
from .Models import TokenEmbeddingModule
import torch.nn.functional as F


class ExplainableModel(nn.Module):
    def __init__(self, embedding_model, freeze_embedding_model=False, num_classes=3, multitask=False, lamb=1.0,
                 multitask_multiplier=1.0):
        """
        This is the main model class for the SelfExplain-Duration model.
        :param embedding_model: The transformer model to use for embedding the input sequence.
        :param freeze_embedding_model: Whether to freeze the weights of the embedding model.
        :param num_classes: The number of classes to predict.
        :param multitask: Whether to use the multitask version of the model (additionally predict duration labels)
        """
        super().__init__()
        self.embedding = TokenEmbeddingModule(embedding_model, freeze_embedding_model)
        self.out_size = self.embedding.out_size
        self.multitask = multitask
        self.span_info_collect = SICModel(self.out_size)
        self.interpretation = InterpretationModel(self.out_size)
        self.standalone_dur_output = nn.Linear(self.out_size, 1)
        self.combined_dur_output = nn.Linear(self.out_size, 1)
        self.output = nn.Linear(self.out_size, num_classes)
        self.multitask_multiplier = multitask_multiplier
        self.lamb = lamb
        # losses
        self.change_loss = nn.CrossEntropyLoss()
        self.duration_loss = nn.MSELoss()

    def forward(self, input_ids, attention_mask, start_indices, end_indices, span_masks, labels=None,
                labels_duration_standalone=None, labels_duration_combined=None, with_explanation=False):
        # generate embeddings using the provided attention mask
        # output.shape = (batch_size, sequence length, hidden_size)
        embeddings = self.embedding(input_ids, attention_mask=attention_mask)
        # span info collecting layer (SIC) creates a representation for each span
        # output.shape = (batch_size, num_spans, hidden_size)
        h_ij = self.span_info_collect(embeddings, start_indices, end_indices)
        # interpretation layer creates attention weights for each span and a weighted sum of the spans for each item
        # output.shape = H: (batch_size, hidden_size); a_ij: (batch_size, num_spans)
        H, a_ij = self.interpretation(h_ij, span_masks)
        # The output layer uses the weighted sum of the spans to predict the label
        out_change = self.output(H)
        out_standalone_dur = None
        out_combined_dur = None
        if self.multitask:
            out_standalone_dur = torch.sigmoid(self.standalone_dur_output(H))
            out_combined_dur = torch.sigmoid(self.combined_dur_output(H))

        if with_explanation:
            # return the most salient span (n=1) for each item
            biggest_span = a_ij.argmax(dim=1)
            # for each item, get the corresponding start and end index
            most_salient_start = start_indices[biggest_span]
            most_salient_end = end_indices[biggest_span]
            return F.softmax(out_change, dim=1), (most_salient_start, most_salient_end)

        if labels is None:
            return F.softmax(out_change, dim=1), None

        loss = self.build_loss(out_change, labels, a_ij, out_standalone_dur, labels_duration_standalone,
                               out_combined_dur, labels_duration_combined, multiplier=self.multitask_multiplier)
        return F.softmax(out_change, dim=1), loss

    def build_loss(self, out_change, labels_change, a_ij, out_standalone=None, labels_standalone=None,
                   out_combined=None, labels_combined=None, multiplier=1.0):
        # calculate the change prediction loss
        loss_change = self.change_loss(out_change, labels_change)
        loss_attention = self.lamb * a_ij.pow(2).sum(dim=1).mean()
        if self.multitask:
            # calculate the duration prediction loss
            loss_standalone_dur = multiplier * self.duration_loss(out_standalone.squeeze(-1), labels_standalone.float())
            loss_combined_dur = multiplier * self.duration_loss(out_combined.squeeze(-1), labels_combined.float())
            return loss_change + loss_standalone_dur + loss_combined_dur + loss_attention
        else:
            return loss_change + loss_attention


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        """
        This module creates a representation of size hidden_size for each span.
        :param hidden_size: The hidden size of the span representations.
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indices, end_indices):
        """
        :param hidden_states: The hidden states of the input sequence (batch_size, length, hidden_size)
        :param start_indices: The start indices of the spans (num_spans)
        :param end_indices: The end indices of the spans (num_spans)
        :return: The hidden representation of each span (batch_size, num_spans, hidden_size)
        """
        # We define weights W1, W2, W3, W4 to be linear transformations of the embeddings
        # output shape: (batch_size, length, hidden_size)
        W1_h = self.W_1(hidden_states)  # Weights for h_i
        W2_h = self.W_2(hidden_states)  # Weights for h_j
        W3_h = self.W_3(hidden_states)  # Weights for h_i - h_j
        W4_h = self.W_4(hidden_states)  # Weights for h_i ⊗ h_j

        # Select the embeddings from the hidden states according to the start and end index
        # output shape: (batch_size, num_spans, hidden_size)
        W1_hi_emb = torch.index_select(W1_h, 1, start_indices)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indices)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indices)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indices)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indices)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indices)

        # sum the vectors up: w1*hi + w2*hj + w3(hi-hj) + w4(hi⊗hj)
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        # apply tanh to get h_ij
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        """
        From hidden representations of spans, we generate attention weights and a re-weighed hidden representation.
        :param hidden_size: The hidden size of the span representations
        """
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        """
        :param h_ij: The hidden representations of all spans (batch_size, num_spans, hidden_size)
        :param span_masks: A large negative number for illegal spans, 0 for legal spans (batch_size, num_spans)
        :return:
        H: The re-weighed hidden representation (batch_size, hidden_size)
        a_ij: The attention weights of the spans (batch_size, num_spans)
        """
        # We transform h_ij (batch_size, num_spans, hidden_size) to (batch_size, num_spans, 1) via a linear layer
        # Then we squeeze the dimension to get a_ij (batch_size, num_spans), i.e. the attention weights
        o_ij = self.h_t(h_ij).squeeze(-1)  # (batch_size, num_spans)
        # mask illegal spans by subtracting by a large negative number
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1 via softmax
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # weight average span representation to get H
        # (batch_size, num_spans, 1) * (batch_size, num_spans, hidden_size) -> (batch_size, num_spans, hidden_size)
        # sum over num_spans to get (batch_size, hidden_size)
        # Effectively, we are creating a weighted average of the span representations for each batch item
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (batch_size, hidden_size)
        # Return attention weights and H (re-weighed hidden representation)
        return H, a_ij
