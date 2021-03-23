import torch

def rec15_collate_fn(data):
    """This function will be used to pad the sessions to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each session (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    
    # padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens

def save_model(model, model_name):
    torch.save(model.state_dict(), './saved_model/'+model_name+'.pickle')

def load_model(model, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./saved_model/'+model_name+'.pickle', map_location=device))

def get_config(model_name, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == 'yoochoose1_64':
        n_items = 37484
        max_seq_length = 19
    elif dataset_name == 'diginetica':
        n_items = 43098
        max_seq_length = 19
    if model_name == 'gru4rec':
        embedding_size = 64
        hidden_size = 128
        num_layers = 1
        dropout_prob = 0.3
        config = {}
        config['device'] = device
        config['n_items'] = n_items
        config['embedding_size'] = embedding_size
        config['hidden_size'] = hidden_size
        config['num_layers'] = num_layers
        config['dropout_prob'] = dropout_prob
        return config
    elif model_name == 'narm':
        embedding_size = 50
        hidden_size = 100
        n_layers = 1
        dropout_probs = [0.25, 0.5]
        config = {}
        config['device'] = device
        config['n_items'] = n_items
        config['embedding_size'] = embedding_size
        config['hidden_size'] = hidden_size
        config['n_layers'] = n_layers
        config['dropout_probs'] = dropout_probs
        return config
    elif model_name == 'sasrec':
        n_layers = 2
        n_heads = 2
        hidden_size = 64
        inner_size = 256
        hidden_dropout_prob = 0.5
        attn_dropout_prob = 0.5
        hidden_act = 'gelu'
        layer_norm_eps = 1e-12
        initializer_range = 0.02
        config = {}
        config['n_items'] = n_items
        config['device'] = device
        config['max_seq_length'] = max_seq_length
        config['n_layers'] = n_layers
        config['n_heads'] = n_heads
        config['hidden_size'] = hidden_size
        config['inner_size'] = inner_size
        config['hidden_dropout_prob'] = hidden_dropout_prob
        config['attn_dropout_prob'] = attn_dropout_prob
        config['hidden_act'] = hidden_act
        config['layer_norm_eps'] = layer_norm_eps
        config['initializer_range'] = initializer_range
        return config
    elif model_name == 'stamp':
        embedding_size = 64
        config = {}
        config['device'] = device
        config['n_items'] = n_items
        config['embedding_size'] = embedding_size
        return config

# add gan's data to train data
def add_data(train, model_name, dataset_name):
    filename = './gan_data/'+model_name+'_'+dataset_name+'.data'
    with open(filename, 'r+', encoding='utf-8') as f:
        gene_data = [i[:-1].split(' ') for i in f.readlines()]
    new_gene_data = []
    for cur_data in gene_data:
        cur_data = [x for x in cur_data if x != '0']
        new_gene_data.append(list(map(int, cur_data)))  
    for line_data in new_gene_data:
        seq_len = len(line_data)
        for idx in range(1,seq_len):
            train[0].append( line_data[0:idx] )
            train[1].append( line_data[idx])
    return train

    
