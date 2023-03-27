import torch
import numpy as np
import torch.optim as optim
import yaml
from tqdm import tqdm

from model.network import MyNet as net
from units.dataloader import MyDataset, DataLoader
from units.weights_init import weights_init
from units.loss import MyLoss as my_loss
import torch


def main():
    # 创建数据加载器
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 构建网络
    model = net(model_layers)
    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 构建损失
    # my_loss = MyLoss
    model = model.cuda()

    # 构建优化器
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)

    # train
    for i in range(num_epoch):
        # train
        train_loss = 0
        model = model.train()

        train_epoch_bar = tqdm(train_dataloader, desc=f"Epoch {i}/{num_epoch}")

        for j_train, batch in enumerate(train_epoch_bar):
            datas, targets = batch[0].cuda(), batch[1].cuda()

            outputs = model(datas)
            loss_batch = my_loss(outputs, targets)
            loss_batch.backward()
            optimizer.step()
            train_loss += loss_batch.item()

            train_epoch_bar.set_postfix(loss=f"{train_loss / (j_train + 1) :.6f}",
                                        current_loss=f"{loss_batch.item():.6f}")

        print("\nTrain Finished")
        # val
        val_loss = 0
        model.eval()

        val_epoch_bar = tqdm(val_dataloader, desc=f"Epoch {i}/{num_epoch}")
        for j_val, batch in enumerate(val_epoch_bar):
            datas, targets = batch[0].cuda(), batch[1].cuda()
            with torch.no_grad():
                outputs = model(datas)
                loss_batch = my_loss(outputs, targets)
            val_loss += loss_batch.item()
            train_epoch_bar.set_postfix(loss=f"{val_loss / (j_val + 1) :.6f}",
                                        current_loss=f"{loss_batch.item():.6f}")

        print(f"\nEpoch_num:{i}, Train_loss:{train_loss / (j_train + 1) :.6f}, Val_loss:{val_loss / (j_val + 1) :.6f}")

        torch.save(model.state_dict(),
                   'logs/ep%03d-loss%.6f-val_loss%.6f.pth' % (i + 1, train_loss / (j_train + 1), val_loss / (j_val + 1)))

    pass


if __name__ == '__main__':
    with open('./config/train_config.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    train_data = train_config["train_data"]
    val_data = train_config["val_data"]
    batch_size = train_config["batch_size"]
    model_path = train_config["model_path"]
    lr = train_config["lr"]
    num_epoch = train_config["num_epoch"]

    model_layers = train_config["model_layers"]

    main()
