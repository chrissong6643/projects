import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    #Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

    import torch.nn as nn

    from .baseRNN import BaseRNN

    class EncoderRNN(BaseRNN):
        r"""
        Applies a multi-layer RNN to an input sequence.
        Args:
            vocab_size (int): size of the vocabulary
            max_len (int): a maximum allowed length for the sequence to be processed
            hidden_size (int): the number of features in the hidden state `h`
            input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
            dropout_p (float, optional): dropout probability for the output sequence (default: 0)
            n_layers (int, optional): number of recurrent layers (default: 1)
            bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
            rnn_cell (str, optional): type of RNN cell (default: gru)
            variable_lengths (bool, optional): if use variable length RNN (default: False)
            embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
                the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
                with the tensor if provided (default: None).
            update_embedding (bool, optional): If the embedding should be updated during training (default: False).
        Inputs: inputs, input_lengths
            - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
            - **input_lengths** (list of int, optional): list that contains the lengths of sequences
                in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        Outputs: output, hidden
            - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
        Examples::
             >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
             >>> output, hidden = encoder(input)
        """

        def __init__(self, vocab_size, max_len, hidden_size,
                     input_dropout_p=0, dropout_p=0,
                     n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                     embedding=None, update_embedding=True):
            super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                             input_dropout_p, dropout_p, n_layers, rnn_cell)

            self.variable_lengths = variable_lengths
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            if embedding is not None:
                self.embedding.weight = nn.Parameter(embedding)
            self.embedding.weight.requires_grad = update_embedding
            self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                     batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        def forward(self, input_var, input_lengths=None):
            """
            Applies a multi-layer RNN to an input sequence.
            Args:
                input_var (batch, seq_len): tensor containing the features of the input sequence.
                input_lengths (list of int, optional): A list that contains the lengths of sequences
                  in the mini-batch
            Returns: output, hidden
                - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
                - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
            """
            embedded = self.embedding(input_var)
            embedded = self.input_dropout(embedded)
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
            output, hidden = self.rnn(embedded)
            if self.variable_lengths:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output, hidden

        import torch
        import torch.nn.functional as F
        from torch.autograd import Variable

        def _inflate(tensor, times, dim):
            """
            Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
            Args:
                tensor: A :class:`Tensor` to inflate
                times: number of repetitions
                dim: axis for inflation (default=0)
            Return Tensor
            """
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)

        class TopKDecoder(torch.nn.Module):
            r"""
            Top-K decoding with beam search.
            Args:
                decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
                k (int): Size of the beam.
            Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
                - **inputs** (seq_len, batch, input_size): list of sequences, whose length is the batch size and within which
                  each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
                - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features
                  in the hidden state `h` of encoder. Used as the initial hidden state of the decoder.
                - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
                  Used for attention mechanism (default is `None`).
                - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
                  (default is `torch.nn.functional.log_softmax`).
                - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
                  drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
                  teacher forcing would be used (default is 0).
            Outputs: decoder_outputs, decoder_hidden, ret_dict
                - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
                  outputs of the decoder.
                - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
                  state of the decoder.
                - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
                  representing lengths of output sequences, *topk_length*: list of integers representing lengths of beam search
                  sequences, *sequence* : list of sequences, where each sequence is a list of predicted token IDs,
                  *topk_sequence* : list of beam search sequences, each beam is a list of token IDs, *inputs* : target
                  outputs if provided for decoding}.
            """

            def __init__(self, decoder_rnn, k):
                super(TopKDecoder, self).__init__()
                self.rnn = decoder_rnn
                self.k = k
                self.hidden_size = self.rnn.hidden_size
                self.V = self.rnn.output_size
                self.SOS = self.rnn.sos_id
                self.EOS = self.rnn.eos_id

            def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax,
                        teacher_forcing_ratio=0, retain_output_probs=True):
                """
                Forward rnn for MAX_LENGTH steps.
                """

                inputs, batch_size, max_length = self.rnn._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                         function, teacher_forcing_ratio)

                self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)

                # Inflate the initial hidden states to be of size: b*k x h
                encoder_hidden = self.rnn._init_state(encoder_hidden)
                if encoder_hidden is None:
                    hidden = None
                else:
                    if isinstance(encoder_hidden, tuple):
                        hidden = tuple([_inflate(h, self.k, 1) for h in encoder_hidden])
                    else:
                        hidden = _inflate(encoder_hidden, self.k, 1)


                if self.rnn.use_attention:
                    inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 0)
                else:
                    inflated_encoder_outputs = None

                # Initialize the scores; for the first step,
                # ignore the inflated copies to avoid duplicate entries in the top k
                sequence_scores = torch.Tensor(batch_size * self.k, 1)
                sequence_scores.fill_(-float('Inf'))
                sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
                sequence_scores = Variable(sequence_scores)

                # Initialize the input vector
                input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))

                # Store decisions for backtracking
                stored_outputs = list()
                stored_scores = list()
                stored_predecessors = list()
                stored_emitted_symbols = list()
                stored_hidden = list()

                for _ in range(0, max_length):

                    # Run the RNN one step forward
                    log_softmax_output, hidden, _ = self.rnn.forward_step(input_var, hidden,
                                                                          inflated_encoder_outputs, function=function)

                    # If doing local backprop (e.g. supervised training), retain the output layer
                    if retain_output_probs:
                        stored_outputs.append(log_softmax_output)

                    # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
                    sequence_scores = _inflate(sequence_scores, self.V, 1)
                    sequence_scores += log_softmax_output.squeeze(1)
                    scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

                    # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
                    input_var = (candidates % self.V).view(batch_size * self.k, 1)
                    sequence_scores = scores.view(batch_size * self.k, 1)

                    # Update fields for next timestep
                    predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(
                        batch_size * self.k, 1)
                    if isinstance(hidden, tuple):
                        hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
                    else:
                        hidden = hidden.index_select(1, predecessors.squeeze())

                    # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
                    stored_scores.append(sequence_scores.clone())
                    eos_indices = input_var.data.eq(self.EOS)
                    if eos_indices.nonzero().dim() > 0:
                        sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

                    # Cache results for backtracking
                    stored_predecessors.append(predecessors)
                    stored_emitted_symbols.append(input_var)
                    stored_hidden.append(hidden)

                # Do backtracking to return the optimal values
                output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                            stored_predecessors, stored_emitted_symbols,
                                                            stored_scores, batch_size, self.hidden_size)

                # Build return objects
                decoder_outputs = [step[:, 0, :] for step in output]
                if isinstance(h_n, tuple):
                    decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
                else:
                    decoder_hidden = h_n[:, :, 0, :]
                metadata = {}
                metadata['inputs'] = inputs
                metadata['output'] = output
                metadata['h_t'] = h_t
                metadata['score'] = s
                metadata['topk_length'] = l
                metadata['topk_sequence'] = p
                metadata['length'] = [seq_len[0] for seq_len in l]
                metadata['sequence'] = [seq[0] for seq in p]
                return decoder_outputs, decoder_hidden, metadata

            def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
                """Backtracks over batch to generate optimal k-sequences.
                Args:
                    nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
                    nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
                    predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
                    symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
                    scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
                    b: Size of the batch
                    hidden_size: Size of the hidden state
                Returns:
                    output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
                    from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
                    h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
                    from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
                    h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
                    score [batch, k]: A list containing the final scores for all top-k sequences
                    length [batch, k]: A list specifying the length of each sequence in the top-k candidates
                    p (batch, k, sequence_len): A Tensor containing predicted sequence
                """

                lstm = isinstance(nw_hidden[0], tuple)
                # initialize return variables given different types
                output = list()
                h_t = list()
                p = list()
                # Placeholder for last hidden state of top-k sequences.
                # If a (top-k) sequence ends early in decoding, `h_n` contains
                # its hidden state when it sees EOS.  Otherwise, `h_n` contains
                # the last hidden state of decoding.
                if lstm:
                    state_size = nw_hidden[0][0].size()
                    h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
                else:
                    h_n = torch.zeros(nw_hidden[0].size())
                l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                # Similar to `h_n`

                sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
                s = sorted_score.clone()

                batch_eos_found = [0] * b  # the number of EOS found
                # in the backward loop below for each batch

                t = self.rnn.max_length - 1
                # initialize the back pointer with the sorted order of the last step beams.
                # add self.pos_index for indexing variable with b*k as the first dimension.
                t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
                while t >= 0:
                    # Re-order the variables with the back pointer
                    current_output = nw_output[t].index_select(0, t_predecessors)
                    if lstm:
                        current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
                    else:
                        current_hidden = nw_hidden[t].index_select(1, t_predecessors)
                    current_symbol = symbols[t].index_select(0, t_predecessors)
                    # Re-order the back pointer of the previous step with the back pointer of
                    # the current step
                    t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()


                    #   For each batch, everytime we see an EOS in the backtracking process,
                    #       1. If there is survived sequences in the return variables, replace
                    #       the one with the lowest survived sequence score with the new ended
                    #       sequences
                    #       2. Otherwise, replace the ended sequence with the lowest sequence
                    #       score with the new ended sequence
                    #
                    eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
                    if eos_indices.dim() > 0:
                        for i in range(eos_indices.size(0) - 1, -1, -1):
                            # Indices of the EOS symbol for both variables
                            # with b*k as the first dimension, and b, k for
                            # the first two dimensions
                            idx = eos_indices[i]
                            b_idx = int(idx[0] / self.k)
                            # The indices of the replacing position
                            # according to the replacement strategy noted above
                            res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                            batch_eos_found[b_idx] += 1
                            res_idx = b_idx * self.k + res_k_idx

                            # Replace the old information in return variables
                            # with the new ended sequence information
                            t_predecessors[res_idx] = predecessors[t][idx[0]]
                            current_output[res_idx, :] = nw_output[t][idx[0], :]
                            if lstm:
                                current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                                current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                                h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                                h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                            else:
                                current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                                h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                            current_symbol[res_idx, :] = symbols[t][idx[0]]
                            s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                            l[b_idx][res_k_idx] = t + 1

                    # record the back tracked results
                    output.append(current_output)
                    h_t.append(current_hidden)
                    p.append(current_symbol)

                    t -= 1

                # Sort and re-order again
                s, re_sorted_idx = s.topk(self.k)
                for b_idx in range(b):
                    l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]]

                re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

                # Reverse the sequences and re-order at the same time
                output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
                p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
                if lstm:
                    h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for
                           step in reversed(h_t)]
                    h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
                else:
                    h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in
                           reversed(h_t)]
                    h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
                s = s.data

                return output, h_t, h_n, s, l, p

            def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
                score[idx] = masking_score

            def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
                if len(idx.size()) > 0:
                    indices = idx[:, 0]
                    tensor.index_fill_(dim, indices, masking_score)



def encode2(pixels):

    #initializing spike train
    train = []

    for l in range(pixels.shape[0]):
        for m in range(pixels.shape[1]):
            temp = np.zeros([(par.T+1),])
            #calculating firing rate proportional to the membrane potential
            freq = interp(pixels[l][m], [0, 255], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pixels[l][m]>0):
                while k<(par.T+1):
                    temp[k] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train

def encode(pot):

    #initializing spike train
    train = []

    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):

            temp = np.zeros([(par.T+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [-1.069,2.781], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pot[l][m]>0):
                while k<(par.T+1):
                    temp[int(k)] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train

if __name__  == '__main__':
    # m = []
    # n = []
    img = imageio.imread("/Users/johnsoni/Downloads/mnist_png/training/5/0.png")
    #img = imageio.imread("data/training/0.png")

    pot = rf(img)

    # for i in pot:
    #     m.append(max(i))
    #     n.append(min(i))

    # print max(m), min(n)
    #train = encode2(img)
    train = encode(pot)
    f = open('train6.txt', 'w')
    print(np.shape(train))

    for j in range(len(train)):
        for i in range(len(train[j])):
            f.write(str(int(train[j][i])))
        f.write('\n')

    f.close()