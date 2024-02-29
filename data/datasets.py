


def data_default(data_dir,flag = 'train'):
    if flag == 'train':
        data = pd.read_csv(os.path.join(data_dir,'train.csv'))
    elif flag == 'val':
        data = pd.read_csv(os.path.join(data_dir,'val.csv'))
    elif flag == 'test':
        data = pd.read_csv(os.path.join(data_dir,'test.csv'))
    else:
        raise ValueError('Invalid flag')
    return data