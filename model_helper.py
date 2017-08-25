import tensorflow as tf

def _single_cell(num_units,forget_bias,dropout):
    single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=forget_bias)
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell,output_keep_prob=(1.-dropout))
    return single_cell

def _cell_list(num_units, num_layers, forget_bias,dropout):
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(num_units=num_units,
                               forget_bias=forget_bias,
                               dropout=dropout)
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(num_units,num_layers,forget_bias,dropout):
    cell_list = _cell_list(num_units,num_layers,forget_bias,dropout)
    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)