import argparse
import os
import shutil
import configparser
from datetime import datetime
import time

import torch
import torch.nn
from torch.utils.data import DataLoader
# import torch.utils.tensorboard
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from lib.data_preparation import read_and_generate_dataset
from lib.data_preparation import unnormalize
from lib.datasets import DatasetPEMS
from model.model_config import get_backbones
from lib.utils import compute_val_loss, evaluate
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()#创建一个ArgumentParser对象,ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息
parser.add_argument("--config", type=str, default='configurations/PEMS04.conf',
                    help="configuration file path", required=False)#给一个ArgumentParser添加程序参数信息是通过调用add_argument()方法完成
#"--config"一个命名或者一个选项字符串的列表,type命令行参数应当被转换成的类型,default当参数未在命令行中出现时使用的值,help此选项作用的简单描述,required表示此命令行选项是否可省略

parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()#ArgumentParser通过parse_args()方法解析参数

# log dir
if os.path.exists('logs'):#查看文件或者文件夹是否存在
    shutil.rmtree('logs')#递归删除文件夹下的所有子文件夹和子文件
    print('Remove log dir')

# read configuration读取配置
#初始化实例
config = configparser.ConfigParser()#定义了一个ConfigParser类
print('Read configuration file: %s' % args.config)
#读取配置文件
config.read(args.config)#num_for_predict = 12:预测12个时间片，每个时间片是5分钟
data_config = config['Data']
training_config = config['Training']

#Data相关赋值
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']#graph_signal_matrix_filename图形信号矩阵文件名
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

#Training相关赋值
model_name = training_config['model_name']
ctx = training_config['ctx']
optimizer = training_config['optimizer']
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
merge = bool(int(training_config['merge']))

# select devices选择设备
if ctx.startswith('cpu'):
    ctx = torch.device("cpu")
    print('cpu')
elif ctx.startswith('gpu'):
    ctx = torch.device("cuda:" + ctx.split('-')[-1])
    print('gpu')
else:
    raise SystemError("error device input")

device = ctx

# import model导入模型
print('Model is %s' % (model_name))
if model_name == 'ASTGCN':
    from model.astgcn import ASTGCN as model
else:
    raise SystemExit('Wrong type of model!')

# make model params dir:将模型参数设为dir
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")#获取当前时间并格式化
if 'params_dir' in training_config and training_config['params_dir'] != "None":
    params_path = os.path.join(training_config['params_dir'], model_name, timestamp)#拼接文件路径
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file:检查参数文件
if os.path.exists(params_path) and not args.force:#os.path.exists(path)如果path存在，返回True；如果path不存在，返回False。
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)#递归删除文件夹下的所有子文件夹和子文件
    os.makedirs(params_path)#创造目录
    print('Create params directory %s' % (params_path))

if __name__ == '__main__':
    start_time = time.perf_counter()#获取当前时间，以秒为单位

    # read all data from graph signal matrix file:从图形信号矩阵文件读取所有数据
    print("Reading data...")
    dataload_start_time = time.perf_counter()#记录数据开始下载的时间
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)#读取并生成数据集
    dataload_end_time = time.perf_counter()#记录数据下载结束的时间
    print(f'Running time for data loading is {dataload_end_time - dataload_start_time:.2f} seconds')

    # test set ground truth:测试集地面真实性
    # true_value = (all_data['test']['target'].transpose((0, 3, 1, 2))
    #               .reshape(all_data['test']['target'].shape[0], -1))
    true_value = (all_data['test']['target'].transpose((0, 3, 1, 2)))


    # training set data loader:训练集数据加载器
    train_dataset = DatasetPEMS(all_data['train'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#将自定义的Dataset根据batch size大小、是否shuffle(打乱)等封装成一个Batch Size大小的Tensor，用于后面的训练。

    # validation set data loader:验证集数据加载器
    val_dataset = DatasetPEMS(all_data['val'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # why shuffle is False?

    # testing set data loader:测试集数据加载器
    test_dataset = DatasetPEMS(all_data['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # save Z-score mean and std
    # stats_data = {}
    # for type_ in ['week', 'day', 'recent']:
    #     stats = all_data['stats'][type_]
    #     stats_data[type_ + '_mean'] = stats['mean']
    #     stats_data[type_ + '_std'] = stats['std']
    #
    # np.savez_compressed(
    #     os.path.join(params_path, 'stats_data'),
    #     **stats_data
    # )

    loss_function = torch.nn.MSELoss()

    all_backbones = get_backbones(args.config, adj_filename, device)
    # print(all_backbones[0][0]['cheb_polynomials'])

    num_of_features = 3
    num_of_timesteps = [[points_per_hour * num_of_weeks, points_per_hour],
                        [points_per_hour * num_of_days, points_per_hour],
                        [points_per_hour * num_of_hours, points_per_hour]]
    net = model(num_for_predict, all_backbones, num_of_vertices, num_of_features, num_of_timesteps, device)#建立神经网络

    net = net.to(device)
    # it is the same as net.to(device)
    # i.e., to() for module is in place, which is different from tensor:to()在张量和模中的位置不同

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)#构建好神经网络后，网络的参数都保存在parameters()函数当中

    # for params in net.parameters():
    #     torch.nn.init.normal_(params, mean=0, std=0.01)

    total_params = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of parameters is: %d" % total_params)
    print("Total number of trainable parameters is: %d" % train_params)

    group_num = 20
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_loss_f = 0.0
        running_loss_o = 0.0
        running_loss_s = 0.0
        epoch_start_time = time.perf_counter()
        batch_start_time = epoch_start_time
        for i, [train_w, train_d, train_r, train_t] in enumerate(train_loader):
            # zero the parameter gradients:使参数梯度归零
            optimizer.zero_grad()

            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            # train_w[:,:,1,:] = train_w[:,:,1,:]*100
            # train_d[:, :, 1, :] = train_d[:, :, 1, :] * 100
            # train_r[:, :, 1, :] = train_r[:, :, 1, :] * 100
            # train_t[:, :, 1, :] = train_t[:, :, 1, :] * 100
            outputs = net([train_w, train_d, train_r])
            # print(type(outputs))
            # print(type(all_data['stats']['target']['mean']))
            outputs = unnormalize(all_data['stats']['stats'],outputs.permute((0, 2, 1, 3)))
            '''
            std=all_data['stats']['target']['std']
            mean=all_data['stats']['target']['mean']
            outp=outputs*std+mean
            
            '''
            max_f=torch.max(outputs[:,:,:,2])
            min_f=torch.min(outputs[:,:,:,2])
            max_t=torch.max(train_t[:,:,2,:])
            min_t = torch.min(train_t[:, :, 2, :])
            loss = loss_function(outputs, train_t.permute((0, 3, 1, 2)))  # loss is a tensor on the same device as outpus and train_t:损耗是一个张量，与outpus和train ut相同
            loss_f = loss_function(outputs[:, :, :, 0], train_t.permute((0, 3, 1, 2))[:, :, :, 0])
            loss_o = loss_function(outputs[:, :, :, 1], train_t.permute((0, 3, 1, 2))[:, :, :, 1])
            loss_s = loss_function(outputs[:, :, :, 2], train_t.permute((0, 3, 1, 2))[:, :, :, 2])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # type of running_loss is float, loss.item() is a float on CPU:运行损失类型为float，loss.item()是CPU上的浮点
            running_loss_f += loss_f.item()
            running_loss_o += loss_o.item()
            running_loss_s += loss_s.item()
            # pytorch中的.item()用于将一个零维张量转换成浮点数
            if i % group_num == group_num - 1:
                batch_end_time = time.perf_counter()
                print(f'[{epoch:d}, {i + 1:5d}] loss: {running_loss / group_num:.2f}, \
                        time: {batch_end_time - batch_start_time:.2f}')
                print(f'f:[{epoch:d}, {i + 1:5d}] loss: {running_loss_f / group_num:.2f}, \
                                        time: {batch_end_time - batch_start_time:.2f}')
                print(f'o:[{epoch:d}, {i + 1:5d}] loss: {running_loss_o / group_num:.2f}, \
                                        time: {batch_end_time - batch_start_time:.2f}')
                print(f's:[{epoch:d}, {i + 1:5d}] loss: {running_loss_s / group_num:.2f}, \
                                        time: {batch_end_time - batch_start_time:.2f}')
                print('--------------------------------------------------------------------')
                running_loss = 0.0
                running_loss_f = 0.0
                running_loss_o = 0.0
                running_loss_s = 0.0
                batch_start_time = batch_end_time

        epoch_end_time = time.perf_counter()
        print(f'Epoch cost {epoch_end_time - epoch_start_time:.2f} seconds')

        # probably not need to run this after every epoch:可能不需要在每一个时代之后都运行这个
        with torch.no_grad():
            # compute validation loss:计算验证损失
            compute_val_loss(net, val_loader, loss_function, None, epoch, device,all_data['stats']['stats'])

            # testing:测试
            evaluate(net, test_loader, true_value, num_of_vertices, None, epoch, device, all_data['stats']['stats'])

    end_time = time.perf_counter()
    print(f'Total running time is {end_time - start_time:.2f} seconds.')






