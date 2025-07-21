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
        model = self.model_dict[self.args.model].Model(self.args).float()

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
