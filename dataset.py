from Bio import SeqIO
import torch
import utils


class MyDataset():
    """构建蛋白质数据集"""

    def __init__(self, fasta_file, shortest_len, longest_len) -> None:
        '''
        初始化每条序列最大长度、fasta文件路径.
            fasta_file: 待训练的fasta数据 
            shortest_len: 最小序列长度
            longest_len: 最大序列长度
            select_method: 'remove': 不将长度超过seq_length的序列加入数据集
                'cut': 将长度超过seq_length的序列右侧多余部分剪掉
        '''
        self.shortest_len = shortest_len
        self.longest_len = longest_len

        self.fasta_file = fasta_file
        self.gap_char = utils.gap_char
        self.spec_char = utils.spec_char
        self.animo_acids = utils.animo_acids
        # self.select_method = select_method
        self.seqs = self.select_seqs()
        self.seq_choice = utils.seq_choice
        self.n_symbols = utils.n_symbols

    def select_seqs(self):
        train_seqs = []

        print('正在筛选长度在 [{}, {}] 的序列...'.format(self.shortest_len, self.longest_len))
        for record in SeqIO.parse(self.fasta_file, 'fasta'):
            if len(record) > self.shortest_len and len(record) <= self.longest_len:
                seq = record.seq
                for _ in range(self.longest_len - len(seq)):
                    seq += self.gap_char
                train_seqs.append(seq)

        print('已筛选出 {} 条符合条件的序列。'.format(len(train_seqs)))
        utils.dataset_size = len(train_seqs)
        return train_seqs

    def seq_2_onehot(self):
        '''将处理后等长的氨基酸序列转独热编码并返回'''
        print('正在将序列转为独热编码...')
        onehot_seqs = []
        for seq in self.seqs:
            # 验证序列长度一致
            assert len(seq) == self.longest_len, '序列长度有误'
            seq_index = []
            for animo_acid in seq:
                # 其余氨基酸按特殊字符'X'处理
                if animo_acid not in self.seq_choice:
                    animo_acid = self.spec_char
                seq_index.append(self.seq_choice.index(animo_acid))
            onehot = [0] * self.longest_len * self.n_symbols
            for i, j in enumerate(seq_index):
                onehot[i * self.n_symbols + j] = 1
            onehot_seqs.append(onehot)
        print('成功。每条序列独热编码后长度为 {}'.format(len(onehot_seqs[0])))
        return onehot_seqs

    def transform_2_tensor(self):
        '''将独热码转为tensor数据类型'''
        train_data = torch.tensor(self.seq_2_onehot(), dtype=torch.float)
        print('独热编码已转为tensor数据类型。\n训练数据处理完毕。')
        return train_data


if __name__ == '__main__':
    fasta_file = "./data/insulin_1_200.fasta"
    max_length = 100
    my_dataset = MyDataset(max_length, fasta_file)
    train_data = my_dataset.transform_2_tensor()
    print(train_data.shape)
    print(train_data.type())
