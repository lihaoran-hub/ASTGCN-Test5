import torch
import torch.nn as nn
import torch.nn.functional as F

debug_on = False
device = None


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores:计算空间注意分数
    """
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Compute spatial attention scores
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Spatial_Attention_layer, self).__init__()

        global device
        self.W_1 = torch.randn(num_of_timesteps, requires_grad=True).to(device)#torch.randn:正态分布随机生成,requires_grad=True表示反向传播
        self.W_2 = torch.randn(num_of_features, num_of_timesteps, requires_grad=True).to(device)
        self.W_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_s = torch.randn(1, num_of_vertices, num_of_vertices, requires_grad=True).to(device)
        self.V_s = torch.randn(num_of_vertices, num_of_vertices, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters:参数
        ----------
        x: tensor, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

           initially, N == num_of_vertices (V)

        Returns
        ----------
        S_normalized: tensor, S', spatial attention scores
                      shape is (batch_size, N, N)

        """
        # get shape of input matrix x:获取输入矩阵x的形状
        # batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # The shape of x could be different for different layer, especially the last two dimensions:对于不同的层，x的形状可能不同，特别是最后两个维度

        # compute spatial attention scores:计算空间注意分数
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)#高维矩阵相乘

        # shape of rhs is (batch_size, T, V)
        # rhs = torch.matmul(self.W_3, x.transpose((2, 0, 3, 1)))
        rhs = torch.matmul(x.permute((0, 3, 1, 2)), self.W_3)  # do we need to do transpose??

        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.V_s, torch.relu(product + self.b_s))

        # normalization:标准化
        S = S - torch.max(S, 1, keepdim=True)[0]#torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引），keepdims主要用于保持矩阵的二维特性
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1, keepdim=True)#keepdim = True表示用[]扩起来
        return F.relu(S_normalized)


class cheb_conv_with_SAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention scores:具有空间注意分数的K阶切比雪夫图卷积
    """

    def __init__(self, num_of_filters, K, cheb_polynomials, num_of_features):
        """
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution:上K-1阶切比雪夫多项式将用于这种卷积

        """
        super(cheb_conv_with_SAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials#切比雪夫多项式

        global device
        self.Theta = torch.randn(self.K, num_of_features, num_of_filters, requires_grad=True).to(device)

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation:切比雪夫图卷积运算

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix:图形信号矩阵
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        """
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        global device

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices,
                                 self.num_of_filters).to(device)  # do we need to set require_grad=True?
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                # theta_k = self.Theta.data()[k]
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                # rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),  # why do we need to transpose?
                #                    graph_signal)
                rhs = torch.matmul(T_k_with_at.permute((0, 2, 1)),
                                   graph_signal)
                #w=torch.matmul(rhs, theta_k)

                output = output + torch.matmul(rhs, theta_k)
            # outputs.append(output.expand_dims(-1))
            outputs.append(torch.unsqueeze(output, -1))#torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
            #torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
            #out=torch.cat(outputs, dim=-1)
        return F.relu(torch.cat(outputs, dim=-1))#torch.cat是将两个张量（tensor）拼接在一起


class Temporal_Attention_layer(nn.Module):
    """
    compute temporal attention scores:计算时间注意力得分
    """

    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Temporal Attention Layer
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Temporal_Attention_layer, self).__init__()

        global device
        self.U_1 = torch.randn(num_of_vertices, requires_grad=True).to(device)
        self.U_2 = torch.randn(num_of_features, num_of_vertices, requires_grad=True).to(device)#如果设置为True，则反向传播时，该tensor就会自动求导
        self.U_3 = torch.randn(num_of_features, requires_grad=True).to(device)#.to(device)表示将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
        self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, x^{(r - 1)}_h
                       shape is (batch_size, V, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: torch.tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        """
        # _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # N == batch_size
        # V == num_of_vertices
        # C == num_of_features
        # T == num_of_timesteps

        # compute temporal attention scores
        # shape of lhs is (N, T, V)
        #a=torch.matmul(x.permute(0, 3, 2, 1), self.U_1)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1),#.permute是tensor的维度换位
                           self.U_2)#torch.matmul是矩阵乘法

        # shape is (batch_size, V, T)
        # rhs = torch.matmul(self.U_3, x.transpose((2, 0, 1, 3)))
        rhs = torch.matmul(x.permute((0, 1, 3, 2)), self.U_3)  # Is it ok to switch the position?

        product = torch.matmul(lhs, rhs)  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        E = torch.matmul(self.V_e, torch.relu(product + self.b_e))

        # normailzation:标准化
        E = E - torch.max(E, 1, keepdim=True)[0]#keepdim（bool）– 保持输出的维度 。当keepdim=False时，输出比输入少一个维度（就是指定的dim求范数的维度）。而keepdim=True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1。
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)

        return F.relu(E_normalized)


class ASTGCN_block(nn.Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps, backbone):
        """
        Parameters
        ----------
        backbone: dict, should have 6 keys,主干：dict，应该有6个键
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",  # wd: never used?? Actually there is no such key in backbone...
                        "time_conv_strides",
                        "cheb_polynomials"
        """
        super(ASTGCN_block, self).__init__()


        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.SAt = Spatial_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)

        self.cheb_conv_SAt = cheb_conv_with_SAt(
            num_of_filters=num_of_chev_filters,
            K=K,
            cheb_polynomials=cheb_polynomials,
            num_of_features=num_of_features)

        self.TAt = Temporal_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)

        self.time_conv = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            stride=(1, time_conv_strides),
            padding=(0, 1))

        self.residual_conv = nn.Conv2d(
            in_channels=num_of_features,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides))


        #self.ln = nn.LayerNorm(num_of_time_filters)#nn.LayerNorm是在channel方向做归一化
        self.ln=nn.LayerNorm(num_of_time_filters)
        self.Liner=torch.nn.Linear(64*12,3*12)
        # self.Liner_o=torch.nn.Linear([64,3])
        # self.Liner_s = torch.nn.Linear([64, 3])

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        torch.tensor, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        """
        #print('block')
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape


        # shape of temporal_At: (batch_size, num_of_timesteps, num_of_timesteps)
        temporal_At = self.TAt(x)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps),#x_Tat是所有节点在T时刻的所有属性
                             temporal_At) \
            .reshape(batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)#batch_size为批次大小，批次大小为64

        # cheb gcn with spatial attention:空间注意cheb-gcn
        # (batch_size, num_of_vertices, num_of_vertices)
        spatial_At = self.SAt(x_TAt)

        # (batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)#空间维卷积

        # convolution along time axis:沿时间轴卷积
        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        time_conv_output = (self.time_conv(spatial_gcn.permute(0, 2, 1, 3))#.permute指的是维度换位,时间维卷积
                            .permute(0, 2, 1, 3))


        # residual shortcut:剩余捷径
        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        x_residual = (self.residual_conv(x.permute(0, 2, 1, 3))
                      .permute(0, 2, 1, 3))

        # (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        #relue = F.relu(x_residual + time_conv_output)#nn.functional.relu只是对relu函数的函数API调用
        #r=x_residual + time_conv_output
        #relued=F.relu(self.last_conv(relue.permute(0,2,1,3)).permute(0,2,1,3))
        #re_conv=self.last_conv(r.permute(0,2,1,3)).permute(0,2,1,3)
        # relued_f=F.relu(re_conv[:,:,0,:])
        # relued_o=F.relu(re_conv[:,:,1,:])
        # relued_s=F.relu(re_conv[:,:,2,:])
        #f=relued[:,:,0,:]
        # r_f=torch.unsqueeze(relued_f,dim=-1)
        # relued_f=torch.squeeze(self.ln(r_f),dim=-1)
        # relued_f = self.ln(relued_f)
        #o = relued[:, :, 1, :]
        # r_o = torch.unsqueeze(relued_o, dim=-1)
        # relued_o = torch.squeeze(self.ln(r_o),dim=-1)
        # relued_o = self.ln(relued_o)
        #s = relued[:, :, 2, :]
        # r_s = torch.unsqueeze(relued_s, dim=-1)
        # relued_s = torch.squeeze(self.ln(r_s),dim=-1)
        #relued_s = self.ln(relued_s)
        # re=torch.stack([relued_f,relued_o,relued_s],dim=3).permute(0, 1, 3, 2)
        #return self.ln(relued.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        relued = F.relu(x_residual + time_conv_output)  # nn.functional.relu只是对relu函数的函数API调用
        r=self.ln(relued.permute(0,1,3,2)).permute(0,1,3,2)
        rel=self.Liner(r.reshape(-1,64*12)).reshape(-1,307,3,12)
        #r=self.ln(relued.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
       # result=self.Liner(r.permute(0,1,3,2)).permute(0,1,3,2)
        #return self.ln(relued.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # return re
        #return result
        return rel


class ASTGCN_submodule(nn.Module):#子模块
    def __init__(self, num_for_prediction, backbones, num_of_vertices, num_of_features, num_of_timesteps):
        """
        submodule to deal with week, day, and hour individually.:子模块分别处理周、日和小时。
        :param num_for_prediction: int
        :param backbones: dict
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: list of int. It includes the num_of_timestep of the input layer, and also of the next layer :int的列表。它包括输入层的时间步长，也包括下一层的
        """
        super(ASTGCN_submodule, self).__init__()


        #all_num_of_features = [num_of_features, backbones[0]["num_of_time_filters"]]
        all_num_of_features = [num_of_features, num_of_features]
        self.blocks = nn.Sequential(*[ASTGCN_block(num_of_vertices,
                                                   all_num_of_features[idx],
                                                   num_of_timesteps[idx],
                                                   backbone)
                                      for idx, backbone in enumerate(backbones)])

        # in_channels: num_of_timestemps, i.e. num_of_weeks/days/hours * args.points_per_hour
        # input(batch, C_in, H_in, W_in)  => output(batch, C_out, H_out, W_out)
        # when padding is 0, and kernel_size[0] is 0, H_out == H_in  --> num_of_vertices
        # out_height = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1
        # out_width = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1
        self.final_conv = nn.Conv2d(in_channels=num_of_timesteps[-1],
                                    out_channels=num_for_prediction,
                                    kernel_size=(1, 1))
        #self.dropout=nn.Dropout(0.2)

        global device
        self.W = torch.randn(num_of_features, num_of_vertices, num_for_prediction, requires_grad=True).to(device)
        self.ler=torch.nn.Linear(307*3,307*3)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor,
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        torch.tensor, shape is (batch_size, num_of_vertices, num_for_prediction)

        """
        #print('sub')
        x = self.blocks(x)

        # the output x's shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)

        # the shape of final_conv()'s output: (batch, num_for_prediction, num_of_vertices, num_of_features_out)
        # module_output = (self.dropout(self.final_conv(x.permute((0, 3, 1, 2)))
        #                  [:, :, :, :].permute((0, 3, 2, 1))))
        module_output = (self.final_conv(x.permute((0, 3, 1, 2)))
                                      [:, :, :, :].permute((0, 3, 2, 1)))
        moudle_f=F.relu(module_output[:,0,:,:])
        moudle_o=F.relu(module_output[:,1,:,:])
        moudle_s=F.relu(module_output[:,2,:,:])
        module=torch.stack([moudle_f,moudle_o,moudle_s],dim=3).permute((0,3,1,2))
      # TODO: why choose the last one of the feature dimension?
        # _, num_of_vertices, num_for_prediction = module_output.shape

        #    (batch_size, num_of_vertices, num_for_prediction)
        #  *             (num_of_vertices, num_for_prediction)
        # => (batch_size, num_of_vertices, num_for_prediction)
        q=module*self.W
        moudle_f = F.relu(q[:, 0, :, :])
        moudle_o = F.relu(q[:, 1, :, :])
        moudle_s = F.relu(q[:, 2, :, :])
        module = torch.stack([moudle_f, moudle_o, moudle_s], dim=3).permute((0, 3, 1, 2))
        module=self.ler(module.permute(0,3,2,1).reshape(-1,307*3)).reshape(-1,3,307,12)
        return module


class ASTGCN(nn.Module):
    def __init__(self, num_for_prediction, all_backbones, num_of_vertices, num_of_features, num_of_timesteps, _device):
        #(self,预测数量,所有主干,顶点数量,特征数量,时间步数,装置)装置指的是用cpu还是gpu
        """
        Parameters
        ----------
        num_for_prediction: int
            how many time steps will be forecasting:将预测多少时间步

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules."hour", "day", "week" 子模块3根主干
                       "week", "day", "hour" (in order)

        num_of_vertices: int
            The number of vertices in the graph:图中顶点的数目

        num_of_features: int
            The number of features of each measurement:每个测量的特征数

        num_of_timesteps: 2D array, shape=(3, 2)
            The timestemps for each time scale (week, day, hour).每个时间刻度（周、日、小时）的TimeTemps。
            Each row is [input_timesteps, output_timesteps].每行都是[input_timesteps, output_timesteps]
        """
        super(ASTGCN, self).__init__()

        global device
        device = _device

        if debug_on:
            print("ASTGCN model:")
            print("num for prediction: ", num_for_prediction)
            print("num of vertices: ", num_of_vertices)
            print("num of features: ", num_of_features)
            print("num of timesteps: ", num_of_timesteps)

        self.submodules = nn.ModuleList([ASTGCN_submodule(num_for_prediction,
                                                          backbones,
                                                          num_of_vertices,
                                                          num_of_features,
                                                          num_of_timesteps[idx])
                                         for idx, backbones in enumerate(all_backbones)])

    def forward(self, x_list):#x_list是[train_w, train_d, train_r]
        """
        Parameters
        ----------
        x_list: list[torch.tensor],
                shape of each element is (batch_size, num_of_vertices,
                                        num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: torch.tensor,
               shape is (batch_size, num_of_vertices, num_for_prediction)
        """
        if debug_on:
            for x in x_list:
                print('Shape of input to the model:', x.shape)

        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = []

        for idx, submodule in enumerate(self.submodules):#enumerate函数可以同时获得索引和值,在此处，控制了backbone和类的对应关系
            #print('ASTGCN:', idx)
            submodule_result = submodule(x_list[idx])
            submodule_result = torch.unsqueeze(submodule_result, dim=-1)#torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度,torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
            submodule_outputs.append(submodule_result)

        submodule_outputs = torch.cat(submodule_outputs, dim=-1)#将submodule_outputs中的张量按照最后一个维度合并到一起

        return torch.sum(submodule_outputs, dim=-1)#dim=-1表示减少一个维度，torch.sum()对输入的tensor数据的某一维度求和
