import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from model import VAE
from dataset import MyDataset
import utils

# 清空日志文件
is_clean = input("是否清空日志文件？(y/n)")
if is_clean == "y":
    utils.clean_logs(utils.VAE_dir,
                     clean_models=True,
                     clean_seqs=True,
                     clean_tb_logs=True)

# 尝试GPU训练
cuda = torch.cuda.is_available()

# 获取数据集
fasta_file = './data/PF00153_RP15/train_data.fasta'

my_dataset = MyDataset(fasta_file, utils.shortest_len, utils.longest_len)
train_data = my_dataset.transform_2_tensor()

# 加载数据集
batch_size = utils.batch_size
dataloader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

# 创建模型
vae = VAE().cuda() if cuda else VAE()
print(vae)

# 设置优化器
optimizer = torch.optim.Adam(vae.parameters(), lr=utils.learning_rate)

# 添加tensorboard
writer = SummaryWriter(utils.VAE_dir + utils.tensorboard_dir)

# ==============================模型训练================================

# 开始计时
start_training = time.time()
start_time = utils.cur_time()
epoch = utils.epoch

for i in range(epoch):
    train_loss = 0
    output = None

    for batch_id, input in enumerate(dataloader):
        if cuda:
            input = input.cuda()

        output, mu, sigma = vae(input)
        loss, output_loss, kl_loss = vae.loss_func(output, input, mu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_id % 20 == 0:
            print(
                "Epoch[{}/{}], Batch[{}/{}], loss:{:.3f}, output_loss:{:.3f}, kl_loss:{:.3f}"
                .format(i + 1, epoch, batch_id, len(dataloader), loss.item(),
                        output_loss.item(), kl_loss.item()))

    print(
        "============================Epoch: {}  loss per batch: {:.3f}============================"
        .format(i + 1, train_loss / batch_id))
    writer.add_scalar("loss_{}".format(utils.protein_name), train_loss, i + 1)

    # 保存生成序列、模型
    if i % 10 == 0:
        seq_save_path = utils.VAE_dir + utils.gen_seqs_dir + "{}_{}_{}.fasta".format(
            utils.protein_name, i, utils.cur_time())
        utils.gen_seq(output, batch_size, seq_save_path)

        if i >= epoch / 2:
            model_save_path = utils.VAE_dir + utils.models_dir + "{}_{}_{}.pth".format(
                utils.protein_name, i, utils.cur_time())
            utils.save_model(vae, model_save_path)

# 结束计时
end_training = time.time()
end_time = utils.cur_time()
cost_time = end_training - start_training

print("训练已完成. \n开始时间：{}\n结束时间：{}\n共用时 {:.3f} 秒.".format(
    start_time, end_time, cost_time))

utils.save_training_logs(utils.VAE_dir)
