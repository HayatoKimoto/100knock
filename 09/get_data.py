def get_data(name):
    label_list = []
    ids_list = []
    x_fname='X_'+name
    y_fname='Y_'+name+'.txt'
    with open(fname) as f:
        lines = f.readlines()
​
    for line in lines:
        if not line:
            continue
        title = line.split('\t')[1].strip('\n')
        word_list = [re.sub(r"^\W|\W$", "", word) for word in title.split()]
        ids = get_ids(word_list)
        ids_list.append(torch.tensor(ids))
        label = category2num[line.split('\t')[0]]
        label_list.append(label)
​
    packed_inputs = pack_sequence(ids_list, enforce_sorted=False)
    padded_packed_inputs = pad_packed_sequence(packed_inputs, batch_first=True)
​
    labels = torch.tensor(label_list)
​
    return padded_packed_inputs, labels