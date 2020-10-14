# -*- coding:utf-8 -*-

import numpy as np

from .utils import get_sample_indices
import torch



def unnormalize(state, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mean=torch.tensor(state['mean']).to(device)
    # std=torch.tensor(state['std']).to(device)
    f_mean=torch.tensor(state['f_mean']).to(device)
    f_std=torch.tensor(state['f_std']).to(device)
    o_mean = torch.tensor(state['o_mean']).to(device)
    o_std = torch.tensor(state['o_std']).to(device)
    s_mean = torch.tensor(state['s_mean']).to(device)
    s_std = torch.tensor(state['s_std']).to(device)
    f_out=output[:,:,0,:]*f_std+f_mean
    o_out = output[:, :, 1, :] * o_std + o_mean
    s_out = output[:, :, 2, :] * s_std + s_mean
    out=torch.stack([f_out,o_out,s_out],dim=0).permute((1,3,2,0))
    # output=output.to(device)
    # return output*std+mean
    return out



def normalization(train, val, test, stats):
    """
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original,形状与原稿相同

    """

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    #mean = train.mean(axis=0, keepdims=True)#keepdims=True保持矩阵维数不变,.mean用来求均值
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)#.std用来计算标准差
    #
    # def normalize(x):
    #     return (x - mean) / std
    #
    # train_norm = normalize(train)  # wd: ??
    # val_norm = normalize(val)
    # test_norm = normalize(test)
    f_mean=stats['f_mean']
    f_std=stats['f_std']
    o_mean=stats['o_mean']
    o_std=stats['o_std']
    s_mean=stats['s_mean']
    s_std=stats['s_std']

    def normalize(x,mmean,sstd):
        # s_mean=np.ones(x.shape)
        # s_mean=s_mean*mmean
        return (x - mmean) / sstd
    train_f_norm=normalize(train[:,:,0,:],f_mean,f_std)
    train_o_norm = normalize(train[:, :, 1, :],o_mean,o_std)
    train_s_norm=normalize(train[:,:,2,:],s_mean,s_std)
    val_f_norm = normalize(val[:, :, 0, :], f_mean, f_std)
    val_o_norm = normalize(val[:, :, 1, :], o_mean, o_std)
    val_s_norm = normalize(val[:, :, 2, :], s_mean, s_std)
    test_f_norm = normalize(test[:, :, 0, :], f_mean, f_std)
    test_o_norm = normalize(test[:, :, 1, :], o_mean, o_std)
    test_s_norm = normalize(test[:, :, 2, :], s_mean, s_std)
    # train_norm=[]
    # val_norm=[]
    # test_norm=[]
    # train_norm.append(train_f_norm)
    # train_norm.append(train_o_norm)
    # train_norm.append(train_s_norm)
    # val_norm.append(val_f_norm)
    # val_norm.append(val_o_norm)
    # val_norm.append(val_s_norm)
    # test_norm.append(test_f_norm)
    # test_norm.append(test_o_norm)
    # test_norm.append(test_s_norm)
    train_norm=np.array([train_f_norm,train_o_norm,train_s_norm]).transpose((1,2,0,3))
    val_norm = np.array([val_f_norm, val_o_norm, val_s_norm]).transpose((1,2,0,3))
    test_norm = np.array([test_f_norm, test_o_norm, test_s_norm]).transpose((1,2,0,3))
    train_norm=train_norm.astype(np.float32)
    val_norm = val_norm.astype(np.float32)
    test_norm=test_norm.astype(np.float32)



    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    """
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 12, depends on data,默认值12

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
         wd: shape is (num_of_samples, num_of_vertices, num_of_features,
                       num_of_weeks/days/hours * points_per_hour)??

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    """
    # wd: if there are 60 days, then 60 * 24 * 12 = 17280, which is close to 16992如果有60天，那么60*24*12=17280，接近16992
    data_seq = np.load(graph_signal_matrix_filename)['data']  # wd: (16992, 307, 3)

    data_seq = np.float32(data_seq)  # wd: to reduce computation,减少计算
    #stats={'f_mean':[],'f_std':[],'o_mean':[],'o_std':[],'s_mean':[],'s_std':[]}
    #stats['f_mean']=np.mean(data_seq[:,:,0])
    data_seq[:,:,1]=data_seq[:,:,1]*100
    f_mean=np.mean(data_seq[:,:,0])
    f_std=np.std(data_seq[:,:,0])
    o_mean = np.mean(data_seq[:, :, 1])
    o_std = np.std(data_seq[:, :, 1])
    s_mean = np.mean(data_seq[:, :, 2])
    s_std = np.std(data_seq[:, :, 2])
    stats = {'f_mean': f_mean, 'f_std': f_std, 'o_mean': o_mean, 'o_std': o_std, 's_mean': s_mean, 's_std': s_std}

    all_samples = []
    for idx in range(data_seq.shape[0]):#shape[0]输出为矩阵的行数,shape[1]输出列数
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),#np.expand_dims:用于扩展数组的形状
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))  # wd: first feature is the traffic flow，第一个特点是交通流,将最后的[:, :, 0, :] 去掉
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        #取出前60%的数据作为训练集的数据
        training_set = [np.concatenate(i, axis=0)#np.concatenate()能够一次完成多个数组的拼接。
                        for i in zip(*all_samples[:split_line1])]#zip是将对象中对应的元素打包成一个个tuple（元组），然后返回由这些 tuples组成的list（列表）

    else:
        print('Merge training set and validation set!')#合并训练集和验证集
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)#获得验证集数据，数据范围是60%到80%之间的20%的数据
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]#获得测试集，测试集是最后20%，即80%到100%

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    # wd: week: (8979, 307, 3, 12), day: (8979, 307, 3, 12), recent: (8979, 307, 3, 36), target: (8979, 307, 12)
    # wd: week/day/recent: num_samples, num_vertices, num_features, num_of_weeks/days/hours * points_per_hour
    # wd: target: num_samples, num_vertices, num_predict
    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week,
                                                    stats)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day,
                                                  stats)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour,
                                                        stats)

    #增加的计算target标准化
    (target_stats,train_target_norm,
     val_target_norm,test_target_norm) = normalization(train_target,
                                                       val_target,
                                                       test_target,
                                                       stats)




    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,  # wd: target does not need to be normalized?
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats,
            'target': target_stats,
            'stats':stats
        }
    }

    return all_data
