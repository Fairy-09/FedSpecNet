# -*- coding: UTF-8 -*-
##LSTM+SSA预处理的预测模型

##和NB_SSA有一个比较明显的不同，就是在数据预处理的时候已经调整成了tensor并且传到了device上
##这个是基于模型的差异进行的一些改动

##number 标记不同node序号不同的部分

import pickle
# import re
import socket
import time
import torch
# from matplotlib.animation import FuncAnimation
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import numpy as np
from support_SSA import SSA 
import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn import tree
# from sklearn import ensemble
# from sklearn.neural_network import MLPRegressor
# from sklearn import svm
# # from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
from sklearn.metrics import mean_squared_error as MSE1
from sklearn.metrics import mean_absolute_error as MAE1
from sklearn import metrics
import ssl

from utils.metrics import metric


# import struct
## LSTM需要引入的部分
class LSTM(nn.Module):
    # def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=1):
    def __init__(self, input_size=1, hidden_size=4, output_size=1, num_layer=1,bidirectional=True):
        super(LSTM, self).__init__()       ##调用父类的构造函数进行初始化
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,bidirectional=bidirectional)   ##布尔值，指示是否使用双向LSTM
        self.layer2 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        ##第二层做了修改的
        ##第一层为LSTM，第二层为全连接层（线性），用于从LSTM的输出生成最终输出

    def forward(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x, _ = self.layer1(x)
        x = torch.relu(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)

        x *= stdev[:, 0, :].unsqueeze(1).repeat(1, 8, 1)
        x += means[:, 0, :].unsqueeze(1).repeat(1, 8, 1)
        return x[:,-1,:]


look_back = 8
EPOCH = 10    ##
head = [None for i in range(look_back)]
SIZE = 8400   ##训练集分割大小
original_mean = 0

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))
def add_noise(parameters):
    parameters = parameters.to(device)  # 确保parameters在正确的设备上
    noise = torch.randn(parameters.shape, device=device).normal_(0, 0.01)  # 在相同的设备上创建噪声张量
    return parameters.add_(noise) # 将噪声添加到参数中并返回
def MAPE1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

## 一、数据预处理

## 分割数据集   改得比较多
def create_dataset(dataset):  ###

    #预测序列和预测结果的分割
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back), 0]) ###9990
        dataY.append(dataset[i + look_back, 0])

    # data_X = np.array(dataX)      ##len=
    # print(np.size(data_X)) #99900
    # data_Y = np.array(dataY)
    # print(np.size(data_Y))#9990
    # ## 训练集和预测集的分割
    # train_size = SIZE - look_back  ##8500-10=8490
    # #test_size = len(dataset) - train_size  ##10000-8490=1510
    # train_X = data_X[:8989]  ##0:8490  8490
    # print(np.size(train_X)) #89890
    # train_Y = data_Y[:8989]
    # print(np.size(train_Y)) #8989
    #
    # # ##这里不需要验证集 去掉
    # # val_X = data_X[train_size:8990]##8490:8990 500
    # # val_Y = data_Y[train_size:8990]
    #
    # val_X = data_X[8989:8990] ##8490:8990 500
    # print(np.size(val_X))  #10
    # val_Y = data_Y[8989:8990]
    # print(np.size(val_Y))  #1
    # test_X = data_X[8990:]         ##8990:9990  1000
    # print(np.size(test_X))  #10000
    # test_Y = data_Y[8990:]
    # print(np.size(test_X))  #10000
    data_X = np.array(dataX)
    data_Y = np.array(dataY)
    train_X = data_X[:7400]
    train_Y = data_Y[:7400]
    val_X = data_X[7400:8400]
    val_Y = data_Y[7400:8400]
    test_X = data_X[8400:12000]
    test_Y = data_Y[8400:12000]
    ## 重塑（训练集/验证集/测试集）
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    train_Y = train_Y.reshape(-1, 1, 1)
    val_X = val_X.reshape(val_X.shape[0], val_X.shape[1], 1)
    val_Y = val_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    test_Y = test_Y.reshape(-1, 1, 1)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y



## coeffs_to_list
def __coeffs_to_list(coeffs, devicce):### [4个(5000*1)np数组]列表
    level_list = []
    for level in range(len(coeffs)):
        var_x, var_y, var_valX, val_Y,var_testX, test_Y = create_dataset(coeffs[level])  ## 对每一层数据进行分割
        ## NB_SSA：level_part = [var_x, var_y, var_valX, val_Y,var_testX, test_Y]
        level_part = [torch.from_numpy(var).float().to(device) for var in [var_x, var_y, var_valX, val_Y, var_testX, test_Y]]
        level_list.append(level_part)
    return level_list    ###4个列表，每个列表里是6个numpy数组

## SSA preprocessing
def __preprocessing_SSA(data, device):   ###data:(5000, 1)
	result_ssa = SSA(data.reshape(-1, ), 4)   ### 转成序列输入，result_ssa:(4, 5000)
	result_ssa = result_ssa.tolist()  ###[[...], [...], [...], [...]]
	for i in range(len(result_ssa)):
		result_ssa[i] = np.array(result_ssa[i]).reshape(-1, 1)  ### [4个(5000*1)np数组]列表
	result_ssa = __coeffs_to_list(result_ssa,device)

	return result_ssa

# load the dataset
def fl_data_load(path, device):
    dataframe = read_csv(path, engine='python')

    dataset = dataframe[0:12000].values        ## ：numpy数组，(5000, 1)
    dataset = dataset.reshape(-1, 1)  ##00## 表示LSTM有NB里面没有的部分
    dataset = dataset.astype('float32')
    np.random.seed(7)

    ##00## scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True) 标准化方式不同
    # dataset = scaler.fit_transform(dataset)
    #scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)  ##00## NB标准化不一样
    dataset = scaler.fit_transform(dataset)

    ## 为去中心化做准备
    original_mean = np.mean(dataset)

    ## 数据读入完成，进行SSA预处理
    data_SSA = __preprocessing_SSA(dataset, device)

    return data_SSA, original_mean  ###4个列表，每个列表里是6个numpy数组



##################################################################################################
## 检查cuda
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

path = r"./dataset/UK-DALE/house1_5min_KWh.csv"    ##number
data_SSA, original_mean = fl_data_load(path, device)

time_steps, input_dim, output_dim = 8, 1, 1
test_size = 3600  ##

# 二、模型构建

model = LSTM(1, 4, 1, 2)   ##修改掉了模型构架的参数
model.to(device)
print(model)
loss_fun = nn.MSELoss()                                       ##均方差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)     ##优化器为Adam

## 注释掉Nbeats部分的代码
# model = LSTMPytorch(backcast_length=time_steps, forecast_length=output_dim,
#                                   stack_types=(NBeatsPytorch.TREND_BLOCK, NBeatsPytorch.SEASONALITY_BLOCK),
#                                   # stack_types=(NBeatsPytorch.GENERIC_BLOCK,NBeatsPytorch.GENERIC_BLOCK),
#                                   nb_blocks_per_stack=3, thetas_dim=(4,4), share_weights_in_stack=True,
#                                   hidden_layer_units=32*4).to(device)
# model.compile(loss='mae', optimizer='rmsprop')

# 三、开始训练
losses = list()
steps = list()

################################# #-# ######################################
for epoch in range(1, EPOCH + 1):
    log("\033[1;31;40m第\033[1;31;40m%s\033[1;31;40m轮开始训练!\033[1;31;40m" % str(epoch))
    # 第一个网络

    ###要在这里改了！！！改成四次循环
    ###############################
    for subSignal_index in range(len(data_SSA)):
        print(str(subSignal_index+1) + '_level SSA data Start.')
#########
##需要改这里，首先不能用fit了，那就用原先的训练代码，
        for t in range(10):   ##每个epoch里面循环10次
            loss_t = list()
            out = model(data_SSA[subSignal_index][0])
            loss = loss_fun(out, data_SSA[subSignal_index][1].squeeze(-1))
            print(loss)
            loss_t.append(loss.item())
            optimizer.zero_grad()           ## 清除优化器的梯度
            loss.backward()                 ## 反向传播
            optimizer.step()                ## 执行一个优化步骤

        losses.append(sum(loss_t)/len(loss_t))
        steps.append(epoch)
        print(str(subSignal_index + 1) + '_level SSA data Complete!!\n')
        ## 绘图
        # plt.rc('font', family='Times New Roman')
        # plt.plot(steps, losses, "o-")
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.draw()
        # plt.pause(0.1)
        # plt.savefig('loss.png')

    log("建立连接并上传......")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 创建SSL上下文
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    # 如果是自签名证书，需要禁用证书验证(不安全，仅用于测试)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # 包装socket为ssl socket
    ssl_socket = context.wrap_socket(s, server_hostname='localhost')

    # 定义中心节点的地址端口号
    host = '127.0.0.1'
    port = 7002

    # 序列化
    data = {}
    data['num'] = (epoch)  ## 这里epoch从0开始
    data['model'] = model.state_dict()

    torch.save(model.state_dict(), 'model.pth')

    keys = model.state_dict().keys()
    data = pickle.dumps(data)  # 将对象序列化obj对象序列化并返回一个byte对象
    # print(s.send(data))
    # 发送数据长度
    data_length = len(data)

    # 建立链接
    try:
        ssl_socket.connect((host, port))
        ssl_socket.sendall(data_length.to_bytes(4, byteorder='big'))
        ssl_socket.sendall(data)
    except Exception as e:
        print(f"error: {e}")

    # 等待接收
    log("等待接收......")
    try:
        ssl_socket.settimeout(100)
        data_length_bytes = ssl_socket.recv(4)
        if not data_length_bytes:
            raise ValueError("未接收到数据长度信息")
        data_length = int.from_bytes(data_length_bytes, 'big')

        # 循环接收直到收到所有数据
        received_data = b''
        while len(received_data) < data_length:
            packet = ssl_socket.recv(data_length - len(received_data))
            if not packet:
                raise ConnectionError("连接中断")
            received_data += packet

        data = pickle.loads(received_data)
        ##
        log(f"收到数据: 轮次 {data['num']}")
        print(data['num'], epoch + 1)
        if data['num'] == epoch + 1:  ## 气死啦！！！因为这里不对齐又吃亏
            global_state_dict = data['model']
            log(f"Update!!!")
        else:
            global_state_dict = model.state_dict()
            log(f"Hold...")
    except Exception as e:
        print(e)
        # s.sendto(data, (host, port))
        log("没有在规定时间收到正确的包， 利用本地参数更新")
        global_state_dict = model.state_dict()

    # global_state_dict = self.model.state_dict()
    # 重新加载全局参数
    model.load_state_dict(global_state_dict)
    s.close()  ## ?
log("训练完毕，关闭连接")
s.close()

plt.plot(steps, losses, "o-")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title('Training and Validation Loss')
plt.legend()
plt.draw()
plt.pause(0.1)
plt.savefig('loss_curve1.png')   # 保存为PNG文件  ##number

plt.close()

################################# #-# ######################################

## 预测并评估
predict_list = []
test_y_list = []
for subSignal_index in range(len(data_SSA)):
    print('No.' + str( subSignal_index+1) + 'Prediction Start.')
    # 预测阶段用 关闭梯度计算
    with torch.no_grad():
        model.eval()

        # Predict on the testing set (forecast).
        predictions_pytorch_forecast = model(data_SSA[subSignal_index][4]).cpu()
        predict = predictions_pytorch_forecast
        predict_list.append(predict)
        test_y_list.append(data_SSA[subSignal_index][5].cpu())
    print('No.' + str( subSignal_index+1) + 'Prediction Complete.')

## 4层预测结果加起来  ##形状不知道对不对 直接复制的 呃
## 将测试集标签和预测值从归一化的形式转换回原始的数值范围。
predict_array = np.stack([tensor.numpy() for tensor in predict_list])  # 这将创建一个形状为 (4, 1000, 1, 1) 的数组
summed_predictions = np.sum(predict_array, axis=0)# 沿着第一个轴（信道轴）对数组进行加和，得到一个 (1000, 1, 1) 的数组
predict = summed_predictions.squeeze()  # 使用 squeeze 方法去除单维度条目
predict = predict + original_mean

test_y_array = np.stack([tensor.numpy() for tensor in test_y_list])
summed_test_y = np.sum(test_y_array, axis=0)
test_y = summed_test_y.squeeze()
test_y = test_y + original_mean

pred_testY_origin = scaler.inverse_transform(predict.reshape(-1, 1))
test_Y_origin = scaler.inverse_transform(test_y.reshape(-1, 1))
###############################



############评估部分############
Dict_Prediction_data = {}
data_test = pd.DataFrame(test_Y_origin)   ##number
data_test.to_csv(r"./myresult/test1.csv", index=False, header=False)
data_pre = pd.DataFrame(pred_testY_origin)
data_pre.to_csv(r"./myresult/pre1.csv", index=False, header=False)

plt.figure()
plt.plot(test_Y_origin, label='Actual', color='blue')
plt.plot(pred_testY_origin, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Value')
plt.savefig('./myresult/res1.png') ##number
# Dict_Prediction_data = {}
# data_test = pd.DataFrame(test_Y_origin)   ##number
# data_test.to_csv(r"./myresult/test1.csv", index=False, header=False)
# data_pre = pd.DataFrame(pred_testY_origin)
# data_pre.to_csv(r"./myresult/pre1.csv", index=False, header=False)
#
# # plt.figure()
# fig=plt.figure(1, figsize=(12, 2.5))
# plt.rc('font', family='Times New Roman', size=16)
# # plt.plot(test_Y_origin, label='Actual', color='blue')
# plt.plot(test_Y_origin, label='Real Value', marker='s',markersize=2)
# plt.plot(pred_testY_origin, label='Predicted Value',linewidth=1,linestyle='-')
# plt.legend()
# plt.xlim(450, 750)
# plt.ylim(0,0.23)
# plt.ylabel('House 1', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # plt.title('Actual vs Predicted')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.savefig('./myresult/res1.png',dpi=500,bbox_inches='tight') ##number

MAE, MSE, RMSE, MAPE, MSPE, R2 = metric(test_Y_origin,pred_testY_origin)

print("MAPE：{:.6f}".format(MAPE))
print("MAE：{:.6f}".format(MAE))
print("MSE：{:.6f}".format(MSE))
print("RMSE：{:.6f}".format(RMSE))
print("R2：{:.6f}".format(R2))

data = pd.DataFrame({'time': range(0, len(test_Y_origin)),
                     'r': test_Y_origin[:, 0],
                     'p': pred_testY_origin[:, 0]})
fig, ax = plt.subplots()


Dict_Prediction = {}
Dict_Prediction['Node1_FLSL_pre'] = pred_testY_origin
np.save('saved\\file' + 'Node1FLSL.npy', Dict_Prediction)


