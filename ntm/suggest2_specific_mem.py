""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
#********************追加Keras*************************
import keras
from keras import backend as K
from keras.models import Sequential,load_model
import time
#************再現性確保*******************************
torch.manual_seed(0)
np.random.seed(0)
#***********終***********************************
import math
count = 0 #追加実装
flag2 = True #追加実装
time_tmp = 0 #追加,時間調査
cls_start = 0
cls_elapsed_time = 0
cs_1000_start = 0
cs_1000_elapsed_time = 0
cs_1000_elapsed_whole_time = 0
id_list = []
repre_id_list = []
std_id_list = [0] * 128
cl_list = []
#***********************************************

class my_algo_fn_specific(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        softmaxの計算
        cosine類似度が1だったらsoftmaxの計算をする。ただしsoftmaxの計算は代表番地とのcosine類似度を使う。
        inputはkeyベクトルと代表ベクトルのcosine類似度の結果が入っている
        """

        # 条件分岐用ので利用するcos_condの計算
        # softmaxを計算すべきクラスのindexを保存しているlistを作成
        # input(cosine類似度)が正の値を取っているところのindexを取得
        cos_cond = np.where(input.data.numpy().reshape(-1)>0)[0].tolist()

        # result用のtensor
        # 最初にinputに対して0より大きいものはそのまま、0より小さいものは0にする
        result = input.clamp(min=0)

        # ここの最大値で引くのはそもそも要素が少ない中最大値を引くと1つのクラスタしかsoftmaxの
        # 計算で残らない可能性があるため、今の所コメントアウトしてある
        # input = input - input.max()
        # cosine類似度が正のクラスタのみでsoftmaxを計算
        result[0, cos_cond] = input[0, cos_cond].exp()/input[0, cos_cond].exp().sum()

        # for debugging ------------------
        # print("cos(input)\n", input)
        # print("cos_cond\n", cos_cond)
        # print("result\n", result)
        # --------------------------------

        ctx.save_for_backward(input, result)

        return result + 1e-8

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, result = ctx.saved_tensors
        # Jのsizeは16*16
        J = torch.zeros(grad_input.size()[-1], grad_input.size()[-1])
        for i in range(grad_input.size()[-1]): 
            for j in range(grad_input.size()[-1]):
                if i==j:
                    J[i][j] = result[0][i]*(1-result[0][i])
                else:
                    J[i][j] = -result[0][i]*result[0][j]

        grad_output = torch.mm(grad_output, J)

        return grad_output


def triplet_loss(y_true, y_pred):
    alpha = 0.3
    out_user, out_pos, out_neg = y_pred[0], y_pred[1], y_pred[2]
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(out_user-out_pos),axis=-1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(out_user-out_neg),axis=-1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

learn_model = load_model('0105_gene_20_67_95_1_109_1.h5', custom_objects={'triplet_loss': triplet_loss})
#learn_model.summary()

def index_mul(l, target):
    return [i for i, _val in enumerate(l) if _val==target]

def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)
        return w


    def _similarity(self, k, β): 
        global count
        global flag2
        global id_list
        global repre_id_list
        global std_id_list
        # time_tmpは全体の経過時間を格納しておく用
        global time_tmp
        # cls_*はclasteringの計算時間用の変数
        global cls_start
        global cls_elapsed_time
        # cs_*はcosine類似度+softmaxの計算時間用の変数
        global cs_1000_start
        global cs_1000_elapsed_time
        global cs_1000_elapsed_whole_time
        
        cs_start = 0
        cs_elapsed_time = 0
        # tmp_cos = [] #代表cos類似度格納
        num_cls = 16

        # print(self.memory.requires_grad)
        # => False
        # print(self.memory)
        print("\nβ requires_grad:", β.requires_grad)
        # => True
        print("β\n", β)
        print("key vector requires_grad", k.requires_grad)
        # => True
        print("key vector\n", k)

        if(flag2):
        # flagはクラスを再編成するかどうか(1000回に1回)
            # id_listはrepre_id_listを作るための一時的なlist --------------
            # clearはただの初期化
            # repre_id_listは各クラスの代表番地を保持するlist
            id_list.clear()
            repre_id_list.clear()
            # -------------------------------------------------------------

            in_count = 0

            # tmp_memoryは1次元に写像して距離計算するために使うmemoryとして利用
            tmp_memory = self.memory[0].data.numpy()
            cls_start = time.time()
            cls_start = time.time()
            # predはmemoryの中身を1次元に写像した結果
            pred = learn_model.predict([tmp_memory, tmp_memory, tmp_memory])
            # tmp_idxはpredの中身に対して昇順に並べ, 元のインデックスのみ保持, numpy array
            tmp_idx = np.argsort(pred[0], axis=0).reshape(-1)
 
            # std_id_listはmemoryの番地に対するクラスラベル labelは16個(0-15)
            # なのでlen()は128
            std_id_list = np.zeros(128, dtype="int8")
            # num_vecは1クラス内のベクトルの個数(今回は8個)
            num_vec = int(tmp_memory.shape[0]//num_cls)
            for i in range(num_cls):
                repre_id_list.append(tmp_idx[int(i*num_vec+num_vec//2 - 1)].tolist())
                std_id_list[tmp_idx[i*num_vec:i*num_vec+num_vec]] = i
            std_id_list = std_id_list.tolist()

            cls_elapsed_time = time.time() - cls_start
            time_tmp += cls_elapsed_time
            flag2 = False
 
            # for debugging ------------------------------------------
            # print("Num-class: ", end = "") #クラス数
            # print(pred[0])
            # print(len(id_list))
            # print(id_list)
            # print("repre_id _list length\n", len(repre_id_list))
            # print("repre_id_list\n", repre_id_list)
            # print("std_id_list\n", std_id_list)
            # print("tmp_idx\n", tmp_idx)
            # --------------------------------------------------------

 
        cs_start = time.time()
        cs_1000_start = time.time()

        # 代表ベクトルを格納
        repre_memory = self.memory[0, repre_id_list]
        # 代表ベクトルとkeyベクトルのcosine類似度を計算
        # suggest2ではceilしない
        cos = β * ((repre_memory+1e-16)*(k+1e-16)).sum(dim=-1)/torch.max((repre_memory+1e-16).norm(dim=-1)*(k+1e-16).norm(dim=-1), torch.tensor([[1e-8]]))
        
        # softmaxの計算
        softmax = my_algo_fn_specific.apply
        cos_after_softmax = softmax(cos)


        # for i in range(128):
        #     if(cl_list[std_id_list[i]] > 0):
        #         # cl_listにはceilで0,1のみにしたcosine類似度がクラス数分(代表番地について)入っている
        #         e_x += np.exp(tmp_cos[std_id_list[i]] - max_val)
        # for i in range(128):
        #     if(cl_list[std_id_list[i]] > 0):
        #         soft_max[0][i] = (np.exp(tmp_cos[std_id_list[i]]- max_val) / e_x)


        w = torch.zeros(1, self.memory.size()[1]) + 1e-8
        for i in range(cos_after_softmax.size()[1]):
            w[0, index_mul(std_id_list, i)] = cos_after_softmax[0, i]


        # for debugging ------------------------------------------
        # print("\nmemory\n", self.memory)
        print("\nrepre memory\n", repre_memory)
        print("\ncos\n", cos)
        print("\ncos after softmax\n", cos_after_softmax)
        print("summation of cos_after_softmax:", cos_after_softmax.sum())
        print("\nw\n", w)
        # --------------------------------------------------------


        cs_elapsed_time = time.time() - cs_start
        cs_1000_elapsed_time += time.time() - cs_1000_start
        time_tmp += cs_elapsed_time

        count += 1

        if(count == 1000):
            cs_1000_elapsed_whole_time += cs_1000_elapsed_time
            print("\npassed time for clastering:" + str(cls_elapsed_time) + "[sec]")  #1回計算したときの時間
            print("passed time for cosine+softmax: " + str(cs_elapsed_time) + "[sec]")  #1回計算したときの時間
            print("passed time for cosine+softmax per 1000: " + str(cs_1000_elapsed_time) + "[sec]")  #1000回計算したときの時間
            print("passed whole_time for cosine+softmax per 1000: " + str(cs_1000_elapsed_whole_time) + "[sec]")  #1000回計算したときの累積時間
            print("passed whole_time for suggest2: " + str(time_tmp) + "[sec]\n")
            flag2 = True
            count = 0
            cs_1000_elapsed_time = 0

        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

