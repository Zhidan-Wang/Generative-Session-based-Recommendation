model_name_set = ['gru4rec', 'narm', 'sasrec', 'stamp']
dataset_name_set = ['yoochoose1_64', 'diginetica']
train_mode_set = ['ori']

for model_name in model_name_set:
    for dataset_name in dataset_name_set:
        for train_mode in train_mode_set:
            cmd = '!python'+' '+'run_seq_model.py'+' '+'--model_name'+' '+model_name+' '+'--dataset_name'+' '+dataset_name+' ' + '--train_mode'+' '+train_mode
            print(cmd)

# gru4rec narm searec stamp