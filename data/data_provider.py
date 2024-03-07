from torch.utils.data import DataLoader
from data.datasets import THU_VibVoltageDataset

def get_data(args,dataset_class = THU_VibVoltageDataset):

    dataset = dataset_class(args.data_dir,flag = 'train')
    train_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args.data_dir,flag = 'val')
    val_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args.data_dir,flag = 'test')
    test_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )     
    return train_loader,val_loader,test_loader