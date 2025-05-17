# !pip install torchviz
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import copy
rng = np.random.default_rng()
np.set_printoptions(precision=4)

# 20240425: 名前が一文字なのは事故の元だったので変更
# 20240511: backpropできるように修正
def next_spike(v, eta, t_inf=1e4, eps=1e-6):
    '''
    compute the next firing time of the neuron.
    v: membrane potential
    eta: intrinsic excitability
    NOTE : epsはもう少し大きくする必要があるかもしれない. ex. eps = 1e-4
    '''
    sqrt_eta = torch.sqrt(torch.abs(eta) + eps)
    v_safe = v + eps * torch.sign(v) + 0.5 * eps * torch.sign(v + eps) # 除算にはv_safeを用いる. 汚いがv = 0でもバグらないはず.
    # v_safe = torch.clamp(torch.abs(v), min = 1e-6) * torch.sign(v) # なぜかこっちはバグる...
    t_inf = torch.tensor(t_inf, device=v.device, dtype=v.dtype)
    return torch.where(eta == 0,
                       torch.where(v > 0, 1/v_safe, t_inf), # eta = 0のときの処理. backward処理を正常に行うためにnp.infではなくt_infを用いる.
                       torch.where(eta < 0,
                                   torch.where(v > sqrt_eta,
                                               torch.atanh(torch.clamp(sqrt_eta / v_safe, -1 + eps, 1 - eps)) / sqrt_eta, # backwardでnanを出さないようにatanhの引数は(-1, 1) でclampする必要がある
                                               t_inf # やはりnp.infを返すとbackwardがうまくいかないのでt_infを返す.
                                              ),
                                   torch.remainder(torch.atan(sqrt_eta/v_safe), torch.tensor(math.pi)) / sqrt_eta
                                  )
                      )

def update_potential(v, eta, dt, eps=1e-6):
    '''
    membrane potential at time t + dt.
    v: membrane potential
    eta: intrinsic excitability
    dt: time to next spike
    NOTE : dtをv, etaとbroadcast可能な形で渡す必要がある.
    '''
    sqrt_eta = torch.sqrt(torch.abs(eta) + eps)

    v = torch.where(eta == 0, v / (1-dt * v), torch.where(eta < 0,
                                                        (v - sqrt_eta * torch.tanh(sqrt_eta * dt)) / (1 - torch.tanh(sqrt_eta * dt) * v / sqrt_eta),
                                                        (v + sqrt_eta * torch.tan(sqrt_eta * dt)) / (1 - torch.tan(sqrt_eta * dt) * v / sqrt_eta)
                                                       )
                    )
    # return torch.clamp(v, vmin, vmax) # ここでclampするとfiring-rate equationとずれる.
    v = torch.clamp(v, -1e25, 1e25) # おまじない
    return v

class QIFNetwork(nn.Module):
    # 20240425 : GPU上でバッチ処理を行うのに適した実装に変更
    def __init__(self, input_size, hidden_size, output_size, params):
        super(QIFNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        try:
            device = params['device'] # そのネットワークに専用のGPUを指定したいときはここで指定する
        except KeyError:
            try:
                device = device # global 変数に指定があればそちらを優先する
            except ValueError:
                device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.i2h = nn.Linear(input_size, hidden_size).to(device)
        self.h2h = nn.Linear(hidden_size, hidden_size).to(device)
        self.h2o = nn.Linear(hidden_size, output_size).to(device)

        # 以下の3つのパラメータは学習しない.
        self.syn_delay = nn.Parameter(torch.tensor(params['syn_delay'], dtype = torch.float).to(device), requires_grad=False)
        self.vmin = nn.Parameter(torch.tensor(params['vmin'], dtype = torch.float).to(device), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(params['vmax'], dtype = torch.float).to(device), requires_grad=False)
    '''
    def initilize_h2h(self, J, eta, delta):
        with torch.no_grad():
            self.h2h.weight.copy_(torch.full(self.h2h.weight.shape, J / self.hidden_size))
            self.h2h.bias.copy_(torch.tensor(rng.standard_cauchy(size = self.hidden_size) * delta + eta))
    '''

    def forward(self, input_spike, input_current, v, t):
        '''
            20240527 :
                - h2h layerを新しく作り, eta, wはそれぞれh2h layerのbias, weightとして表現した.
                - input, outputともにspikeとcurrent (連続量) 両方を受け取れるように変更した.
            inputs : 
                input_spike : spike input to hidden neuron.
                input_current : current input to hidden neuron.
                v : membrane potential of (hidden) neurons. 
                t : time.
            outputs : 
                output_spike : synaptic output to the next layer.
                output_potential : output as weighted sum of membrane potential.
                v : updated membrane potential of (hidden) neurons.
                t : updated time.
                dt : time to the next spike.
                k : index of the neuron that spikes.                
            TODO : 2種類のoutputに同じ重み (self.h2o.weight) を使っているが, 使い分ける必要があるかもしれない.
            NOTE : kから投射する重みを利用するにはw.T[k]と転置する必要がある. Linear layerの内部的な実装でも転置されている.
        '''
        # assert v.dtype == self.h2h.weight.dtype, print("input dtype is different from network dtype ", v.dtype, self.h2h.weight.dtype)
        assert v.dtype == t.dtype == input_current.dtype == input_spike.dtype, print("dtype is different among inputs ", v.dtype, t.dtype, input_current.dtype, input_spike.dtype)
        BATCHSIZE = v.shape[0] 
        assert input_spike.shape[0] == input_current.shape[0] == v.shape[0] == t.shape[0] == BATCHSIZE

        v = v + input_spike.view(BATCHSIZE, self.hidden_size) # add input spikes

        eta = self.h2h.bias + self.i2h(input_current.view(BATCHSIZE, self.input_size)).view(BATCHSIZE, -1) # add input current
        dt_all = next_spike(v, eta)
        assert dt_all.shape == (BATCHSIZE, self.hidden_size)
        min_dt, k = torch.min(dt_all, 1)
        assert min_dt.shape[0] == k.shape[0] == BATCHSIZE

        with torch.no_grad():
            syn_delay = self.syn_delay
            # もしかするとsyn_delayをadaptiveにしたほうがよいかもしれない. ここではとりあえず固定しておく.
            # syn_delay = torch.max(next_spike(torch.ones_like(eta) * self.vmax, eta), 1)[0] + self.syn_delay 
        
        v = update_potential(v, eta, (min_dt + syn_delay).view(-1, 1))
        assert v.shape == (BATCHSIZE, self.hidden_size)
        # TODO: DenseH2H classを作成し, 場合分けをなくす.
        # --- ① self.h2h.col(k) が無い (= 普通の nn.Linear) ときの救済 ---
        if hasattr(self.h2h, 'col'):
            v = v + self.h2h.col(k)                           # KronH2H や DenseH2H
        else:                                                 # ふつうの nn.Linear
            # self.h2h.weight.t() : (hidden, hidden) → (B, hidden)
            v = v + self.h2h.weight.t()[k]                    # k は (B,) なのでバッチ毎に列を取れる
        t = t + min_dt + syn_delay
        assert t.shape[0] == BATCHSIZE
        output_spike = self.h2o.weight.T[k]
        output_potential = self.h2o(torch.clamp(v, self.vmin, self.vmax)).view(BATCHSIZE, -1) # v自体はclampせず, outputだけclampする.
        return output_spike, output_potential, v, t, min_dt, k

def resample_time_series(x_rec, t_rec, t_arr):
    '''
    Resample time series data to the given time bins.
    x_rec: (T, B, D) array, where B is the number of samples, T is the number of time steps, and D is the dimension of the observed data.
    t_rec: (T, B) array, where B is the number of samples, T is the number of time steps.
    t_arr: the time bins to resample the data.
    return: (len(t_arr), D) array, where D is the dimension of the observed data.
    '''
    D = x_rec.shape[-1] # dimension of the observed data
    x_rec.reshape(-1, D)
    t_rec.reshape(-1, 1)
    assert x_rec.shape[0] == t_rec.shape[0]
    x_arr = [] 

    for i in range(len(t_arr)):
        start_time = t_arr[i]
        end_time = t_arr[i + 1] if i < len(t_arr) - 1 else t_arr[i] + t_arr[i] - t_arr[i - 1] # 最後の時間binの場合は、前のbinの幅を使う

        # 時間binに含まれる観測データを抽出
        mask = (t_rec >= start_time) & (t_rec < end_time)
        x_in_bin = x_rec[mask]

        # 該当する観測データが存在する場合、その平均を計算
        if len(x_in_bin) > 0:
            x_arr.append(np.mean(x_in_bin, axis=0))
        else:
            # 観測データが存在しない場合は、NaNを追加
            x_arr.append(np.full(D, np.nan))
    x_arr = np.array(x_arr)
    return x_arr

class KronH2H(torch.nn.Module):
    """
    h = (w_fre ⊗ 1_{N×N} / N) @ x
    ・重み本体（P×P）は w_fre だけ保持
    ・eta（元々 h2h.bias に入っていた乱数）は bias に残す
    """
    def __init__(self, w_fre: np.ndarray, N: int, eta_init: torch.Tensor):
        super().__init__()
        P = w_fre.shape[0]
        self.P, self.N = P, N
        self.register_buffer('w_fre', torch.tensor(w_fre, dtype=torch.float32))
        self.bias = torch.nn.Parameter(eta_init, requires_grad=False)  # shape = (P*N,)

    # --- 行列積 (ほぼ使わないが互換性のため実装) ---
    def forward(self, x):                        # x: (B, P*N)
        B = x.size(0)
        x = x.view(B, self.P, self.N)            # → (B, P, N)
        g_mean = x.mean(dim=2)                   # (B, P)
        y_g    = g_mean @ self.w_fre.T           # (B, P)
        y      = y_g.unsqueeze(2).expand(-1, -1, self.N) / self.N
        return y.reshape(B, -1) + self.bias      # (B, P*N)

    # --- 「列を 1 本だけ」欲しいときに使う ---
    def col(self, k: torch.LongTensor):
        """
        k: (B,)  スパイクしたニューロン index
        戻り値: (B, P*N)  ＝ weight.T[k]
        """
        g = (k // self.N)                        # (B,) presynaptic group
        # w_fre[:, g] → (P, B) → (B, P)
        w_pg = self.w_fre[:, g].T                # gather 1 列だけ
        w_pg = w_pg.unsqueeze(2).expand(-1, -1, self.N) / self.N
        return w_pg.reshape(k.size(0), self.P * self.N)
