# -*- coding:utf-8 -*-
# pylint: disable=no-member

import csv
import numpy as np
from scipy.sparse.linalg import eigs
import torch
from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np
# from .data_preparation import unnormalize
def unnormalize(state, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mean=torch.tensor(state['mean']).to(device)
    # std=torch.tensor(state['std']).to(device)
    f_mean = torch.tensor(state['f_mean']).to(device)
    f_std = torch.tensor(state['f_std']).to(device)
    o_mean = torch.tensor(state['o_mean']).to(device)
    o_std = torch.tensor(state['o_std']).to(device)
    s_mean = torch.tensor(state['s_mean']).to(device)
    s_std = torch.tensor(state['s_std']).to(device)
    f_out = output[:, :, 0, :] * f_std + f_mean
    o_out = output[:, :, 1, :] * o_std + o_mean
    s_out = output[:, :, 2, :] * s_std + s_mean
    out = torch.stack([f_out, o_out, s_out], dim=0).permute((1, 3, 2, 0))
    # output=output.to(device)
    # return output*std+mean
    return out
def unormalize(state, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mean=torch.tensor(state['mean']).to(device)
    # std=torch.tensor(state['std']).to(device)
    f_mean = torch.tensor(state['f_mean']).to(device)
    f_std = torch.tensor(state['f_std']).to(device)
    o_mean = torch.tensor(state['o_mean']).to(device)
    o_std = torch.tensor(state['o_std']).to(device)
    s_mean = torch.tensor(state['s_mean']).to(device)
    s_std = torch.tensor(state['s_std']).to(device)
    output=torch.tensor(output).to(device)
    f_out = output[:, 0, :, :] * f_std + f_mean
    o_out = output[:, 1, :, :] * o_std + o_mean
    s_out = output[:, 2, :, :] * s_std + s_mean
    out = torch.stack([f_out, o_out, s_out], dim=0).permute((1, 3, 2, 0))
    # output=output.to(device)
    # return output*std+mean
    return out
def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data,所有历史数据的长度

    num_of_batches: int, the number of batches will be used for training,批数将用于培训

    label_start_idx: int, the first index of predicting target,预测目标的第一个指标

    num_for_predict: int,
                     the number of points will be predicted for each sample,将预测每个样本的点数

    units: int, week: 7 * 24, day: 24, recent(hour): 1，3个类分别对应的时间长度

    points_per_hour: int, number of points per hour, depends on data,每小时点数,取决于数据

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i#points_per_hour是一个小时内有12个片，units表示有多少个小时，i表示取得的是第几周或第几天的数据
        end_idx = start_idx + num_for_predict  # wd: this could overlap with 'label_start_index', e.g. when num_for_predict is larger than 12 (one hour)，这可能与'label_start_index'重叠，例如 当num_for_predict大于12（一小时）时
        #end_idx表示的是当前节点后12个时间片的3个特征
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):#label_start_idx可以理解为当前所在的时间点，即从当前时间点开始，往前找相应时间片的3个特征，往后确定要预测的12个时间片的3个特征
    """
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)形状是（序列长度，顶点数，特征数）

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target预测目标的第一个指标

    num_for_predict: int,
                     the number of points will be predicted for each sample将预测每个样本的点数

    points_per_hour: int, default 12, number of points per hour每小时点数

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,  # wd: points_per_hour should be num_for_predict??
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    """
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)#总共取了24个时间片，也就是两个小时，分别是上一周这个时间点的前一个小时和上上周这个时间点的前一个小时的3个特征
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)#总共取了12个时间片，也就是一个小时，取得的是昨天这个时间点的前一个小时的3个特征
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)#总共取了24个时间片，也就是2个小时，即当前时间点的前两个小时的3个特征
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i: j]#np.concatenate 是numpy中对array进行拼接的函数
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]#在此获得要预测的后12个时间片的数据

    return week_sample, day_sample, hour_sample, target


def get_adjacency_matrix(distance_df_filename, num_of_vertices):#求邻接矩阵
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information,csv文件的路径包含边缘信息

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))#np.diag的作用是当array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵,array是一个二维矩阵时，结果输出矩阵的对角线元素

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real#eigs获取最大特征值,此处指的是获得拉普拉斯矩阵的最大特征值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):#切比雪夫多项式
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}计算从T_0到T{K-1}的chebyshev多项式列表

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N),比例拉普拉斯算子

    K: the maximum order of chebyshev polynomials,切比雪夫多项式的最大阶

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]#np.identity(N):生成一个N阶单位矩阵

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, sw, epoch, device, state):#计算验证集的损失值
    """
    compute mean loss on validation set:计算验证集上的平均损失

    Parameters
    ----------
    net: model

    val_loader: DataLoader

    loss_function: func

    sw: SummaryWriter. TODO: to be implemented

    epoch: int, current epoch

    """
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
        val_w = val_w.to(device)
        val_d = val_d.to(device)
        val_r = val_r.to(device)
        val_t = val_t.to(device)
        output = net([val_w, val_d, val_r])
        output = unnormalize(state,output.permute((0, 2, 1, 3)))
        l = loss_function(output, val_t.permute((0, 3, 1, 2)))  # l is a tensor, with single value,l是张量，具有单值
        # l_f = loss_function(output.permute((0, 3, 1, 2))[:,:,:,0], val_t.permute((0, 3, 1, 2))[:,:,:,0])
        # l_o = loss_function(output.permute((0, 3, 1, 2))[:,:,:,1], val_t.permute((0, 3, 1, 2))[:,:,:,1])
        # l_s = loss_function(output.permute((0, 3, 1, 2))[:,:,:,2], val_t.permute((0, 3, 1, 2))[:,:,:,2])
        tmp.append(l.item())#pytorch中的.item()用于将一个零维张量转换成浮点数
        print('validation batch %s / %s, loss: %.2f' % (
            index + 1, val_loader_length, l.item()))
        # print('f:validation batch %s / %s, loss: %.2f' % (
        #     index + 1, val_loader_length, l_f.item()))
        # print('o:validation batch %s / %s, loss: %.2f' % (
        #     index + 1, val_loader_length, l_o.item()))
        # print('s:validation batch %s / %s, loss: %.2f' % (
        #     index + 1, val_loader_length, l_s.item()))

    validation_loss = sum(tmp) / len(tmp)

    if sw:
        sw.add_scalar(tag='validation_loss',
                    value=validation_loss,
                    global_step=epoch)

    print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))


def predict(net, test_loader, device):#预测
    """
    predict

    Parameters
    ----------
    net: model

    test_loader: DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    """

    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r, _) in enumerate(test_loader):
        test_w = test_w.to(device)
        test_d = test_d.to(device)
        test_r = test_r.to(device)
        prediction.append(net([test_w, test_d, test_r]).cpu().numpy())
        print('predicting testing set batch %s / %s' % (index + 1, test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction


def evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch, device, state):
    """
    compute MAE, RMSE, MAPE scores of the prediction:计算预测的MAE、RMSE、MAPE得分
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: DataLoader

    true_value: np.ndarray, all ground truth of testing set,测试装置的全部真实性
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    sw: SummaryWriter. TODO: to be implemented.

    epoch: int, current epoch

    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction = predict(net, test_loader, device)
    prediction = unormalize(state,prediction)
    #prediction = prediction.reshape(prediction.shape[0], -1).data.cpu().numpy()
    prediction = prediction.data.cpu().numpy()
    for i in [3, 6, 12]:
        print('current epoch: %s, predict %s points' % (epoch, i))

        mae = mean_absolute_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices])
        rmse = mean_squared_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices]) ** 0.5
        mape = masked_mape_np(true_value[:, : i * num_of_vertices],
                              prediction[:, : i * num_of_vertices], 0)
        mae_f = mean_absolute_error(true_value[:, : i ,:,0],
                                  prediction[:, : i ,:,0])
        rmse_f = mean_squared_error(true_value[:, : i ,:,0],
                                  prediction[:, : i ,:,0]) ** 0.5
        mape_f = masked_mape_np(true_value[:, : i ,:,0],
                              prediction[:, : i ,:,0], 0)
        mae_o = mean_absolute_error(true_value[:, : i, :, 1],
                                    prediction[:, : i, :, 1])
        rmse_o = mean_squared_error(true_value[:, : i, :, 1],
                                    prediction[:, : i, :, 1]) ** 0.5
        mape_o = masked_mape_np(true_value[:, : i, :, 1],
                                prediction[:, : i, :, 1], 0)
        mae_s = mean_absolute_error(true_value[:, : i, :, 2],
                                    prediction[:, : i, :, 2])
        rmse_s = mean_squared_error(true_value[:, : i, :, 2],
                                    prediction[:, : i, :, 2]) ** 0.5
        mape_s = masked_mape_np(true_value[:, : i, :, 2],
                                prediction[:, : i, :, 2], 0)

        # print('MAE: %.2f' % (mae))
        # print('RMSE: %.2f' % (rmse))
        # print('MAPE: %.2f' % (mape))
        print('f:MAE: %.2f' % (mae_f))
        print('f:RMSE: %.2f' % (rmse_f))
        print('f:MAPE: %.2f' % (mape_f))
        print()
        print('o:MAE: %.2f' % (mae_o))
        print('o:RMSE: %.2f' % (rmse_o))
        print('o:MAPE: %.2f' % (mape_o))
        print()
        print('s:MAE: %.2f' % (mae_s))
        print('s:RMSE: %.2f' % (rmse_s))
        print('s:MAPE: %.2f' % (mape_s))
        print()
        if sw:
            sw.add_scalar(tag='MAE_%s_points' % (i),
                        value=mae,
                        global_step=epoch)
            sw.add_scalar(tag='RMSE_%s_points' % (i),
                        value=rmse,
                        global_step=epoch)
            sw.add_scalar(tag='MAPE_%s_points' % (i),
                        value=mape,
                        global_step=epoch)
