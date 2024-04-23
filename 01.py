import torch
from torch import nn
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import Accuracy


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./minist/",
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = T.Compose([T.ToTensor()])
        self.ds_test = MNIST(self.data_dir, train=False, transform=transform, download=True)
        self.ds_predict = MNIST(self.data_dir, train=False, transform=transform, download=True)
        ds_full = MNIST(self.data_dir, train=True, transform=transform, download=True)
        self.ds_train, self.ds_val = random_split(ds_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.ds_predict, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


data_mnist = MNISTDataModule()
data_mnist.setup()

for features, labels in data_mnist.train_dataloader():
    print(features.shape)
    print(labels.shape)
    break

torch.Size([32, 1, 28, 28])
torch.Size([32])

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.1),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)


class Model(pl.LightningModule):

    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.train_acc = Accuracy(num_classes=10, average='macro', multiclass=True, task='multiclass')
        self.val_acc = Accuracy(num_classes=10, average='macro', multiclass=True, task='multiclass')
        self.test_acc = Accuracy(num_classes=10, average='macro', multiclass=True, task='multiclass')

    def forward(self, x):
        x = self.net(x)
        return x

    # 定义loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    # 定义各种 metrics
    def training_step_end(self, outputs):
        train_acc = self.train_acc(outputs['preds'], outputs['y']).item()
        self.log("train_acc", train_acc, prog_bar=True)
        return {"loss": outputs["loss"].mean()}

    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    def validation_step_end(self, outputs):
        val_acc = self.val_acc(outputs['preds'], outputs['y']).item()
        self.log("val_loss", outputs["loss"].mean(), on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    def test_step_end(self, outputs):
        test_acc = self.test_acc(outputs['preds'], outputs['y']).item()
        self.log("test_acc", test_acc, on_epoch=True, on_step=False)
        self.log("test_loss", outputs["loss"].mean(), on_epoch=True, on_step=False)


model = Model(net)

# 查看模型大小
model_size = pl.utilities.memory.get_model_size_mb(model)
print("model_size = {} M \n".format(model_size))
model.example_input_array = [features]
summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
print(summary)

pl.seed_everything(1234)

ckpt_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min'
)
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            mode='min')

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练

trainer = pl.Trainer(max_epochs=20,
                     # gpus=0, #单CPU模式
                     gpus=-1,  # 单GPU模式
                     # num_processes=4,strategy="ddp_find_unused_parameters_false", #多CPU(进程)模式
                     # gpus=[0,1,2,3],strategy="dp", #多GPU的DataParallel(速度提升效果一般)
                     # gpus=[0,1,2,3],strategy=“ddp_find_unused_parameters_false" #多GPU的DistributedDataParallel(速度提升效果好)
                     callbacks=[ckpt_callback, early_stopping],
                     profiler="simple")

# 断点续训
# trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

# 训练模型
trainer.fit(model, data_mnist)

# 评估
result = trainer.test(model, data_mnist.train_dataloader(), ckpt_path='best')

# 使用
data,label = next(iter(data_mnist.test_dataloader()))
model.eval()
prediction = model(data)
print(prediction)


# 保存模型
print(trainer.checkpoint_callback.best_model_path)
print(trainer.checkpoint_callback.best_model_score)

# 输出结果
model_clone = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trainer_clone = pl.Trainer(max_epochs=3, gpus=1)
result = trainer_clone.test(model_clone, data_mnist.test_dataloader())
print(result)