# 使用 WGAN 模型生成氨基酸序列

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from model import Generator, Discriminator
from dataset import MyDataset
import utils

# 清空日志文件
is_clean = input("是否清空日志文件？(y/n)")
if is_clean == "y":
    utils.clean_logs(utils.GAN_dir,
                     clean_models=True,
                     clean_seqs=True,
                     clean_tb_logs=True)

# 尝试GPU训练
cuda = torch.cuda.is_available()

# 获取数据集
fasta_file = "./data/PF00658_FULL/PF00658_full.fasta"

my_dataset = MyDataset(fasta_file, utils.shortest_len, utils.longest_len)
train_data = my_dataset.transform_2_tensor()

# 加载数据集
batch_size = utils.batch_size
dataloader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

# 创建模型
generator = Generator().cuda() if cuda else Generator()
discriminator = Discriminator().cuda() if cuda else Discriminator()
print("生成器模型架构：\n{}\n".format(generator))
print("判别器模型架构：\n{}\n".format(discriminator))

# 设置生成器迭代步长（每训练多少batch的生成器，训练一次判别器）
n_critic = 5
# 判别器梯度截断范围边界值 (WGAN-GP使用GP代替截断策略)
# clip_value = 1e-2
# Loss weight for gradient penalty
lambda_gp = 10

# 设置优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=utils.learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=utils.learning_rate)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 添加tensorboard
writer = SummaryWriter(utils.GAN_dir + utils.tensorboard_dir)


def get_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples +
                    ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0).detach()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty


# ==============================模型训练================================

# 开始计时
start_training = time.time()
start_time = utils.cur_time()
epoch = utils.epoch

for i in range(epoch):

    train_loss_G = 0
    train_loss_D = 0

    for batch_id, input in enumerate(dataloader):
        if cuda:
            input = input.cuda()

        # 真实序列
        real_seqs = input

        # ------------训练判别器（D）------------
        optimizer_D.zero_grad()

        # 生成随机噪音，作为生成器输入
        mu, sigma = 0, 1
        z = Tensor(np.random.normal(mu, sigma, (input.shape[0], 100)))
        # print(input.shape[0])
        # if cuda:
        #     z = z.cuda()

        # 生成序列
        fake_seqs = generator(z)

        # Gradient penalty
        gradient_penalty = get_gradient_penalty(discriminator, real_seqs.data,
                                                fake_seqs.data)

        # 计算损失
        loss_D = -torch.mean(discriminator(real_seqs)) + torch.mean(
            discriminator(fake_seqs)) + lambda_gp * gradient_penalty

        loss_D.backward()
        optimizer_D.step()

        train_loss_D += loss_D

        # 对判别器进行梯度截断
        # for p in discriminator.parameters():
        #     p.data.clamp_(-clip_value, clip_value)

        # ------------训练生成器（G）------------
        if batch_id % n_critic == 0:  # 每5个batch，更新一次生成器的权重参数
            optimizer_G.zero_grad()
            # 生成序列
            gen_seqs = generator(z)
            # 计算损失
            loss_G = -torch.mean(discriminator(gen_seqs))

            loss_G.backward()
            optimizer_G.step()

            train_loss_G += loss_G

        if batch_id % 20 == 0:
            print("Epoch[{}/{}], Batch[{}/{}], loss_G:{:.3f}, loss_D:{:.3f}".
                  format(i + 1, epoch, batch_id, len(dataloader),
                         loss_G.item(), loss_D.item()))

    print(
        "===================Epoch: {}   loss G: {:.3f}   loss D: {:.3f}====================="
        .format(i + 1, train_loss_G, train_loss_D))
    writer.add_scalar("loss_G_{}".format(utils.protein_name), train_loss_G,
                      i + 1)
    writer.add_scalar("loss_D_{}".format(utils.protein_name), train_loss_D,
                      i + 1)

    # 保存生成序列、模型（生成器G）
    if (i + 1) % 20 == 0:
        seq_save_path = utils.GAN_dir + utils.gen_seqs_dir + "{}_{}_{}.fasta".format(
            utils.protein_name, i + 1, utils.cur_time())
        utils.gen_seq(gen_seqs, batch_size, seq_save_path, cut_gap=False)

        model_save_path = utils.GAN_dir + utils.models_dir + "{}_{}_{}.pth".format(
            utils.protein_name, i + 1, utils.cur_time())
        utils.save_model(generator, model_save_path)

# 结束计时
end_training = time.time()
end_time = utils.cur_time()
cost_time = end_training - start_training

print("训练已完成. \n开始时间：{}\n结束时间：{}\n共用时 {:.3f} 秒.".format(
    start_time, end_time, cost_time))

utils.save_training_logs(utils.GAN_dir)
