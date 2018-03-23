# LSTM XOR solver
This script trains an LSTM to solve the XOR problem (determining the parity of a sequence of bits)

The architecture used is a simple 1-layer LSTM with a hidden state of size 32. The LSTM cell used is the default (no peepholes) Tensorflow lstm cell: `tf.nn.rnn_cell.LSTMCell`, which is itself based on [Hochreiter and Schmidhuber, 1997][1]. The optimizer used is Adam

### Usage
`python xor_solver.py <sequence_data_filename> <sequence_length_data_filename> <mode> <model_dir>`
Arguments:

* `sequence_data_filename` - name of the file containing your binary sequences. The file should be of type `.npy` and should contain a single numpy array of type `int` (and elements only 0 or 1) and dimensions `(num_sequences, max_sequence_length`
* `sequence_length_data_filename` - name of the file containing the lengths of your binary sequences. The file should be of type `.npy` and should contain a single numpy array of type `int` (and elements between 1 and `max_sequence_length`) and dimension `num_sequences`
* `mode` - set to `train` if you want to train the network, `eval` if you want to evaluate its performance.
* `model_dir` - directory where the model checkpoints will be written to / read from

Note that the script automatically splits your data in a training and validation set.

[1]: http://www.bioinf.jku.at/publications/older/2604.pdf
