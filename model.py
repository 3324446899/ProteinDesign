"""VAE、GAN模型定义"""
import torch
from torch import nn
import torch.nn.functional as F
import utils


# ===========================VAE==============================================

class VAE(nn.Module):
    '''VAE模型'''

    def __init__(self) -> None:
        super().__init__()

        # 输入数据大小（22*100=2200）
        self.input_size = utils.input_size

        # -----编码器-----
        self.encoder = nn.Sequential(
            # layer 0
            nn.Linear(self.input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # layer 1
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # layer 3-μ
        self.mu = nn.Linear(128, 16)
        # layer 3-σ
        self.log_var = nn.Sequential(
            nn.Linear(128, 16),
            nn.Softplus(),
        )

        # -----解码器-----
        self.decoder = nn.Sequential(
            # layer 4
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # layer 5
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # layer 6
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # layer 7
            nn.Linear(512, self.input_size),
            nn.Sigmoid(),
        )

    def reparameterization(self, mu, log_var):
        '''对p(z)（高斯分布）进行随机采样，返回采样结果 (重参数化技巧)'''
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z

    def forward(self, x):
        # 编码器
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        z = self.reparameterization(mu, log_var)
        # 解码器
        output = self.decoder(z)
        return output, mu, log_var

    def loss_func(self, output, x, mu, log_var):
        '''损失函数'''
        output_loss = F.mse_loss(output, x, reduction="sum")
        if torch.cuda.is_available():
            output_loss = output_loss.cuda()
        kl_loss = 0.5 * (torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
        kl_loss = torch.sum(kl_loss)
        loss = output_loss + kl_loss
        return loss, output_loss, kl_loss


# ================================GAN===============================================

class Generator(nn.Module):
    """GAN-generator"""
    def __init__(self, in_feat=100) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # layer 0
            nn.Linear(in_feat, 128),
            # nn.BatchNorm1d(128), # WGAN-GP中此操作与grandient penalty冲突，故去掉
            nn.LeakyReLU(),
            # layer 1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # layer 2
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # layer 3
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            # layer 4
            nn.Linear(1024, utils.input_size),
            nn.Tanh(),
        )
    
    def forward(self, z):
        seq_one_hot = self.model(z)
        return seq_one_hot


class Discriminator(nn.Module):
    """GAN-discriminator"""
    def __init__(self) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            # layer 0
            nn.Linear(utils.input_size, 512),
            nn.LeakyReLU(),
            # layer 1
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            # layer 2
            nn.Linear(256, 1),
            # nn.Sigmoid(), # WGAN去掉了Sigmoid()函数
        )
    
    def forward(self, input):
        validity = self.model(input)
        return validity


if __name__ == '__main__':

    # vae = VAE()
    # input = torch.ones((utils.batch_size, utils.input_size))
    # output, mu, log_var = vae(input)

    # print(output.shape)
    # print(mu.shape)
    # print(log_var.shape)
    generator = Generator()
    input = torch.ones((utils.batch_size, 100))
    output = generator(input)
    print(output.shape)

    discriminator = Discriminator()
    out2 = discriminator(output)
    print(out2.shape)
