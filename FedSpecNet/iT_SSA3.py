# -*- coding: UTF-8 -*-

import pickle
import socket
import time
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import argparse

from support_SSA import SSA 
import numpy as np

from sklearn.metrics import mean_squared_error as MSE1
from sklearn.metrics import mean_absolute_error as MAE1
from sklearn import metrics


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
import pywt
import math
warnings.filterwarnings('ignore')



# import struct
## LSTM需要引入的部分
# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_size=4, output_size=1, num_layer=1,bidirectional=True):
#         super(LSTM, self).__init__()       ##调用父类的构造函数进行初始化
#         self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,bidirectional=bidirectional)   ##布尔值，指示是否使用双向LSTM
#         self.layer2 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
#         ##第二层做了修改的
#         ##第一层为LSTM，第二层为全连接层（线性），用于从LSTM的输出生成最终输出

#     def forward(self, x):
#         x, _ = self.layer1(x)
#         x = torch.relu(x)
#         s, b, h = x.size()
#         x = x.view(s * b, h)
#         x = self.layer2(x)
#         x = x.view(s, b, -1)
#         return x[:,-1,:]

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))
def MAPE1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#exp def
def find_next_valid_value(data, start_index, direction='forward'):

    if direction == 'forward':
        for i in range(start_index + 1, len(data)):
            if not math.isnan(data[i]) and data[i] > 0:
                return data[i]
    elif direction == 'backward':
        for i in range(start_index - 1, -1, -1):
            if not math.isnan(data[i]) and data[i] > 0:
                return data[i]
    return None
def iswt_decom(data, wavefunc):
    y = data[0]
    for i in range(len(data) - 1):
        y = pywt.iswt([(y, data[i + 1])], wavefunc)
    return y

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        ## 增加使得self.model.state_dict()可以被调用
        self.model = self._build_model()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,raw_label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_time = time.time()
        for epoch in range(self.args.train_epochs):
            print(f"Epoch {epoch + 1}/{self.args.train_epochs}")
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,raw_label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) 

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            ## 加入fed连接的部分
            train_loss = np.average(train_loss)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            log("建立连接并上传......")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 定义中心节点的地址端口号
            host = '127.0.0.1'
            port = 7002
            # 建立链接
            s.connect((host, port))
            # 序列化
            data = {}
            data['num'] = (epoch + 1)  ## 这里epoch从0开始
            data['model'] = self.model.state_dict()

            torch.save(self.model.state_dict(), 'model.pth')

            keys = self.model.state_dict().keys()
            data = pickle.dumps(data)     # 将对象序列化obj对象序列化并返回一个byte对象
            #print(s.send(data))
            # 发送数据长度
            data_length = len(data)
            s.sendall(data_length.to_bytes(4, byteorder='big'))

            # 发送实际数据
            s.sendall(data)

            # 等待接收
            log("等待接收......")
            try:
                s.settimeout(100)
                data_length_bytes = s.recv(4)
                if not data_length_bytes:
                    raise ValueError("未接收到数据长度信息")
                data_length = int.from_bytes(data_length_bytes, 'big')

                # 循环接收直到收到所有数据
                received_data = b''
                while len(received_data) < data_length:
                    packet = s.recv(data_length - len(received_data))
                    if not packet:
                        raise ConnectionError("连接中断")
                    received_data += packet

                data = pickle.loads(received_data)
                ##
                log(f"收到数据: 轮次 {data['num']}")
                print(data['num'], epoch+1)
                if data['num'] == epoch+1:  ## 气死啦！！！因为这里不对齐又吃亏
                    global_state_dict = data['model']
                    log(f"Update!!!")
                else:
                    global_state_dict = self.model.state_dict()
                    log(f"Hold...")
            except Exception as e:
                print(e)
                # s.sendto(data, (host, port))
                log("没有在规定时间收到正确的包， 利用本地参数更新")
                global_state_dict = self.model.state_dict()

            #global_state_dict = self.model.state_dict()
            # 重新加载全局参数
            self.model.load_state_dict(global_state_dict)
            s.close()   ## ?
        log("训练完毕，关闭连接")
        s.close()
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test') 
        preds = []
        trues = []
        raws = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,raw_label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse: 
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                if(self.args.decomposition_method!="swt"):
                    preds.append(np.sum(pred.squeeze()))
                    trues.append(np.sum(true.squeeze()))
                else:
                    preds.append(pred.squeeze())
                    trues.append(true.squeeze())
                raws.append(raw_label.item())

        raws_1 = raws
        if (self.args.decomposition_method != "swt"):

            preds_1 = np.array(preds).reshape(-1, 1)
            trues_1 = np.array(trues).reshape(-1, 1)

            preds_2 = test_data.inverse_transform(preds_1)
            trues_2 = test_data.inverse_transform(trues_1)

        else:
            preds2 = np.array(np.array(preds).T.tolist())
            true2 = np.array(np.array(trues).T.tolist())
            wavefun = pywt.Wavelet('db1')
            preds_1 = iswt_decom(preds2, wavefun)
            trues_1 = iswt_decom(true2, wavefun)

            preds_3 = np.array(preds_1).reshape(-1, 1)
            trues_3 = np.array(trues_1).reshape(-1, 1)


            preds_2 = test_data.inverse_transform(preds_3)
            trues_2 = test_data.inverse_transform(trues_3)

        mae, mse, rmse, mape, mspe, R2 = metric(preds_2, trues_2)

        for j in range(len(preds_2)):
            if math.isnan(preds_2[j]) or preds_2[j] <= 0:  
                if j == 0:  
                    replacement = find_next_valid_value(preds_2, j, 'forward')
                elif j == len(preds_2) - 1:  
                    replacement = find_next_valid_value(preds_2, j, 'backward')
                else:  
                    next_valid = find_next_valid_value(preds_2, j, 'forward')
                    prev_valid = find_next_valid_value(preds_2, j, 'backward')
                    if next_valid is not None and prev_valid is not None:
                        replacement = (next_valid + prev_valid) / 2
                    elif next_valid is not None:
                        replacement = next_valid
                    elif prev_valid is not None:
                        replacement = prev_valid
                    else:
                        replacement = 0  
                if replacement is not None:
                    preds_2[j] = replacement
                else:
                    preds_2[j] = 0  
        mae, mse, rmse, mape, mspe, R2 = metric(preds_2, trues_2)

        true_first = trues_2.flatten()[1]
        pred_first = preds_2.flatten()[1]

        trues_2_new = trues_2[2:]
        preds_2_new = preds_2[2:]

        data = [{'TRUES': true, 'PREDS': pred} for true, pred in zip(trues_2_new.flatten(), preds_2_new.flatten())]
        row_data = [{'TRUES': true_first, 'PREDS': pred_first,'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe, 'R2': R2}]
        print('mse:{}, mae:{},rmse:{},mape:{},mspe:{},R2:{}'.format(mse, mae, rmse, mape, mspe, R2))


        cvs_name = test_data.model_id
        csv_file_path = f'./experiment_result/result/predict-truth/{cvs_name}.csv'


        write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0
        a = row_data
        b = data
        with open(csv_file_path, 'a', newline='') as csvfile:
            fieldnames = ['TRUES', 'PREDS','MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'R2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(row_data)
            writer.writerows(data)
        return


# import struct of iTransformer
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    # iTransformer
    parser.add_argument('--decomposition_method', type=str, required=True, default="ssa", help='ssa/vmf/swt/ewt/emd')
    parser.add_argument('--subsequence_num', type=int, required=True, default=4, help='ssa subsequence num')
    parser.add_argument('--interval', type=int, required=True, default=1, help='time step(1,2,4,6,12,means 5min,10min,20min,30min,60min)')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet,iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='UK-DALE', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/UK-DALE/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='house1_5min_KWh.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=8, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') #!
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')#!
    parser.add_argument('--c_out', type=int, default=7, help='output size')#!
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')#!
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads') #8
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') #!
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')#!
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]') #timeF表示使用固定的频率对数据进行处理，这里的频率是h  --freq为h
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10 , help='train epochs') #10
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#0.0001
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPUS
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    args.ild = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_dm{}_sl{}_ll{}_pl{}_{}_sm{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.decomposition_method,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.interval,
                args.subsequence_num,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            # import ipdb; ipdb.set_trace()
            exp.train(setting)

            exp.test(setting)
            print("\n ")
            torch.cuda.empty_cache()
        print(args.model_id,"  DONE!!!")


# #######原先模型一些参数设置########
# look_back = 8
# EPOCH = 16    ##
# head = [None for i in range(look_back)]
# SIZE = 8400   ##训练集分割大小
# original_mean = 0
# ##################################





