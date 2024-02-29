from cgi import test
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


from torch import seed
from trainer.trainer_basic import TSPN_trainer
from config import args
from config import signal_processing_modules,feature_extractor_modules

#data
from data.data_provider import get_data


seed_everything(17)
name = f'num{args.num}_model{args.model}_dataset{args.dataset}_dim{args.dim}_depth{args.depth}\
    _task_list{args.task_list}_lr{args.lr}_mask{args.mask}_epoches{args.epoches}_seed{args.seed}/'
path = 'save/' + name
# 初始化模型
model = TSPN_trainer(signal_processing_modules, feature_extractor_modules, args)

# 设置检查点回调以保存模型
checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    dirpath = path
)

# 初始化训练器
trainer = pl.Trainer(callbacks=[checkpoint_callback],
                     max_epochs=args.num_epochs,
                     logger=CSVLogger(path))

# dataset
train_dataloader, val_dataloader, test_dataloader = get_data(args)

# train
trainer.fit(model,train_dataloader, val_dataloader)

# 加载最佳模型
best_model_path = checkpoint_callback.best_model_path
best_model = model.load_from_checkpoint(best_model_path)

# 使用最佳模型进行测试
trainer.test(best_model, dataloaders=test_dataloader)

# # 假设test_dataloader是测试集的DataLoader
# y_true = []
# y_pred = []

# best_model.eval()
# with torch.no_grad():
#     for x, y in test_dataloader:
#         logits = best_model(x)
#         preds = torch.argmax(logits, dim=1)
#         y_true.extend(y.tolist())
#         y_pred.extend(preds.tolist())

# # 绘制混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# sns.heatmap(cm, annot=True)
# plt.show()

# # 计算准确率
# accuracy = accuracy_score(y_true, y_pred)
# print(f'Test Accuracy: {accuracy}')

# # 保存准确率到CSV
# df = pd.DataFrame({"Accuracy": [accuracy]})
# df.to_csv(f"{path}/accuracy.csv", index=False)