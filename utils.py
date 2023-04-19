"""
一些辅助函数、定义
"""
import os
import shutil
import torch
import datetime

# 训练参数
batch_size = 32
learning_rate = 0.001
epoch = 500
dataset_size = None
# 加入数据集的最短序列长度
shortest_len = 0
# 加入数据集的最长序列长度
longest_len = 75
# 蛋白质种类/名称
protein_name = 'PF00658_FULL'
# 补齐字符
gap_char = '-'
# 特殊氨基酸字符
spec_char = 'X'
# 20种氨基酸
animo_acids = [
    'G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S', 'P', 'V', 'I', 'C', 'Y',
    'H', 'R', 'N', 'D', 'T'
]
seq_choice = animo_acids + [spec_char, gap_char]
# 氨基酸、替代符号的总个数
n_symbols = len(seq_choice)
# VAE模型输入大小/GAN生成器输出大小
input_size = longest_len * n_symbols

# 模型目录
GAN_dir = "./GAN"
VAE_dir = "./VAE"
# 日志目录
tensorboard_dir = "/tensorboard_logs/"
models_dir = "/results/models/"
gen_seqs_dir = "/results/seqs/"
test_results_dir = "/results/test_results/"
good_models_dir = "/good_models/"
training_logs_dir = "/training_logs/"


def gen_seq(onehot, n_proteins, save_path, write_mode="w", cut_gap=True):
    '''将独热码转为氨基酸序列'''
    # assert len(onehot) == seq_length * n_symbols * n_proteins, '独热编码长度有误'
    # onehot size: 32x2200
    onehot = onehot.reshape(-1)
    seqs = ""
    for i in range(n_proteins * longest_len):
        animo_acid_idx = torch.argmax(onehot[i * n_symbols:(i + 1) *
                                             n_symbols])
        seqs += seq_choice[animo_acid_idx.item()]

    # 将总的序列切分为一个个蛋白质序列存入文件
    with open(save_path, write_mode) as file_obj:
        for i in range(n_proteins):
            seq = seqs[i * longest_len:(i + 1) * longest_len]
            if cut_gap:
                seq = seq.replace("-", "")
            # 写入文件
            file_obj.write(">{}_{}\n".format(i + 1, cur_time()))
            file_obj.write(seq + '\n')

    print("生成序列已保存至 " + save_path)


def save_model(model, path):
    '''保存模型'''
    torch.save(model.state_dict(), path)
    print("模型已保存至'{}'".format(path))


def cur_time():
    '''获取当前时间，返回格式化字符串'''
    t = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return t


def clean_logs(model,
               clean_models=False,
               clean_seqs=False,
               clean_tb_logs=False):
    '''清空results/models, results/seqs, logs_train目录'''
    if clean_models:
        shutil.rmtree(model + models_dir)
        os.mkdir(model + models_dir)

    if clean_seqs:
        shutil.rmtree(model + gen_seqs_dir)
        os.mkdir(model + gen_seqs_dir)

    if clean_tb_logs:
        shutil.rmtree(model + tensorboard_dir)
        os.mkdir(model + tensorboard_dir)


def save_training_logs(model):
    '''保存训练过程信息'''
    save_path = model + training_logs_dir + "log_{}.txt".format(protein_name)
    log_str = cur_time() + "\n"
    log_str += "seq length:     [{}, {}]\n".format(shortest_len, longest_len)
    log_str += "dataset size:   {}\n".format(dataset_size)
    log_str += "batch size:     {}\n".format(batch_size)
    log_str += "learning rate:  {}\n".format(learning_rate)
    log_str += "total epoch:    {}\n".format(epoch)
    log_str += "----------------------------------------------------------------\n"
    with open(save_path, "a") as file_obj:
        file_obj.write(log_str)
    print("训练日志已保存。")


if __name__ == "__main__":
    c1 = input('clean tb_logs? press enter to continue... ')
    c2 = input('clean models? press enter to continue... ')
    c3 = input('clean seqs? press enter to continue... ')
    clean_logs(c1, c2, c3)
    save_training_logs()