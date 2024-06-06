import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

from model.aug import sim_global

########################################
## Spatial Heterogeneity Modeling
########################################
class SpatialHeteroModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        
        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)
        
        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

########################################
## Temporal Heterogeneity Modeling
########################################
class TemporalHeteroModel(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''
    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalHeteroModel, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda':
            self.lbl = lbl.cuda()
        
        self.n = batch_size

    def forward(self, z1, z2):
        '''
        :param z1, z2 (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1) # nlvc->nvc
        s = self.read(h) # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim)
        :return s: summary, (batch_size, feat_dim)
        '''
        s = torch.mean(h, dim=1)
        s = self.sigm(s) 
        return s

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1) # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2) 
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits

####################
## ST Encoder
####################
### self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                       ### input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
import numpy as np
import os
class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, graph_init, learnable_flag, droprate=0.1):
        super(STEncoder, self).__init__()        
        
        self.do_sconv = True
        if graph_init == "no_sconv":
            self.do_sconv = False

        self.learnable_flag = learnable_flag

        if input_length - 2 * (Kt - 1) * len(blocks) <= 0:
            self.Ks=Ks
            c = blocks[0]
            self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU", paddin='valid', flag=False)
            # self.represent = representationLayer(Kt, 1, c[1], "GLU", paddin='valid', flag=False)
            self.pooler = Pooler(input_length - (Kt - 1), c[1])
            
        
            self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
            t = input_length + 2 - 2 - 2 
            # self.sconv12 = SpatialAttention(d_model=c[1], n_timesteps=t, n_heads=1)
            
            self.tconv13 = TemporalConvLayer(Kt, c[1], c[2], paddin='same', flag=True)
            self.ln1 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout1 = nn.Dropout(droprate)

            c = blocks[1]
            self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU", paddin='same', flag=True)
            
            self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
            t = input_length + 2 - 2 - 2 - 2 - 2 
            # self.sconv22 = SpatialAttention(d_model=c[1], n_timesteps=t, n_heads=1)
            
            self.tconv23 = TemporalConvLayer(Kt, c[1], c[2], paddin='same', flag=True)
            self.ln2 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout2 = nn.Dropout(droprate)
            
            self.s_sim_mx = None
            self.t_sim_mx = None

            out_len = input_length - (Kt - 1) # input_length - 8
            self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
            self.ln3 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout3 = nn.Dropout(droprate)
            self.receptive_field = input_length + Kt -1

        else:
            self.Ks=Ks
            c = blocks[0]
            self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
            # self.represent = representationLayer(Kt, 1, c[1], "GLU", paddin='valid', flag=False)
            self.pooler = Pooler(input_length - (Kt - 1), c[1])
            
            self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
            t = input_length + 2 - 2 - 2 
            # self.sconv12 = SpatialAttention(d_model=c[1], n_timesteps=t, n_heads=1)
            self.lns1 = nn.LayerNorm([num_nodes, c[1]])
            self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
            self.ln1 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout1 = nn.Dropout(droprate)

            c = blocks[1]
            self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
            
            self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
            t = input_length + 2 - 2 - 2 - 2 - 2 
            # self.sconv22 = SpatialAttention(d_model=c[1], n_timesteps=t, n_heads=1)
            self.lns2 = nn.LayerNorm([num_nodes, c[1]])
            self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
            self.ln2 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout2 = nn.Dropout(droprate)
            
            self.s_sim_mx = None
            self.t_sim_mx = None

            out_len = input_length - 2 * (Kt - 1) * len(blocks)   # input_length - 8
            self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
            self.ln3 = nn.LayerNorm([num_nodes, c[2]])
            self.dropout3 = nn.Dropout(droprate)
            self.receptive_field = input_length + Kt -1

            self.learnable_flag = learnable_flag
        

    def forward(self, x0, learnable_graph):
        # print("x0.shape: ", x0.shape)
        # print("graph.shape: ", graph.shape)
        if self.learnable_flag == False:
            lap_mx = self._cal_laplacian(learnable_graph)      ## from adj to laplacian
            Lk = self._cheb_polynomial(lap_mx, self.Ks)
        elif self.learnable_flag == True:
            # Lk = graph.unsqueeze(0)
            Lk = learnable_graph
        # # print("Lk.shape: ", Lk.shape)
        # Lk = graph.unsqueeze(0)

        ## if stacking learnbale and og mats
        # lap_mx = self._cal_laplacian(graph)      ## from adj to laplacian
        # Lk = self._cheb_polynomial(lap_mx, self.Ks)
        # Lk = torch.cat((Lk, learnable_graph), dim=0)
        in_len = x0.size(1) # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0,0,0,0,self.receptive_field-in_len,0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes), nclv 
        
        ## ST block 1
        
        """lets work here, as this is the start of embedding, so a bottleneck here would limit the performance downstream."""
        
        # print("x.shape (before tconv11): ", x.shape)  torch.Size([32, 2, 37, 200])
        x = self.tconv11(x)    # nclv          
        # x = self.represent(x)    # nclv          
        # print("x.shape (after represent): ", x.shape)   ## torch.Size([32, 32, 35, 200])
        
        "...end..."
        
        x, x_agg, self.t_sim_mx = self.pooler(x)
        # print("x.shape (after pooler): ", x.shape)   torch.Size([32, 32, 33, 200])
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')
        # print("x.shape (before sconv12): ", x.shape)  torch.Size([32, 32, 33, 200])
        if self.do_sconv:
            # x_skip = x
            print("x.shape: ", x.shape, "Lk.shape: ", Lk.shape)
            x = self.sconv12(x, Lk)   # nclv
            # x = x + x_skip
            ## [b, c, t, n]
            x = self.lns1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     ## ln([b, t, n, c]) -> [b, c, t, n]
        # print("x.shape (after sconv12): ", x.shape)  torch.Size([32, 32, 33, 200])
        x = self.tconv13(x)  
        # print("x.shape (after tconv13): ", x.shape)    torch.Size([32, 64, 31, 200])
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        ## ST block 2
        x = self.tconv21(x)
        # print("x.shape (after tconv21): ", x.shape)  torch.Size([32, 32, 29, 200])
        # print("x.shape (before sconv22): ", x.shape)  torch.Size([32, 32, 29, 200])
        if self.do_sconv:
            # x_skip = x
            x = self.sconv22(x, Lk)   # nclv
            # x = x + x_skip
            x = self.lns2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # print("x.shape (after sconv22): ", x.shape)  torch.Size([32, 32, 29, 200])
        x = self.tconv23(x)
        # print("x.shape (after tconv23): ", x.shape)  torch.Size([32, 64, 27, 200])
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        # print("x.shape: (before out_conv)", x.shape)   torch.Size([32, 64, 27, 200]) 
        # print("\n\n out_conv next: ")
        x = self.out_conv(x) # ncl(=1)v    ## filter_size = (l, 1), so dot product with time length and same kernel used for every node.
        # print("\n\nout_conv done\n\n")
        # print("x.shape: (after out_conv)", x.shape)   torch.Size([32, 64, 1, 200])
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1))) # nlvc
        return x # nl(=1)vc

    def _cheb_polynomial(self, laplacian, K):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [v, v].
        :return: the multi order Chebyshev laplacian, [K, v, v].
        """
        # print("approximating cheb_polynomial of order {}".format(K))
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float) 
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - multi_order_laplacian[k-2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        """
        return the laplacian of the graph.

        :param graph: the graph structure **without** self loop, [v, v].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I # add self-loop to prevent zero in D
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        # print("self.c_in: ", self.c_in, "self.c_out: ", self.c_out)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu", paddin='valid', flag=False):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.flag = flag
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1, padding=paddin)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1, padding=paddin)

    def forward(self, x):
        """
        :param x: (n,c,l,v)
        :return: (n,c,l-kt+1,v)
        """
        # print("x.shape (before align): ", x.shape)
        # print("kt: ", self.kt)
        # print("x.shape (before align): ", x.shape)
        if self.flag:
            x_in = self.align(x)  
        else:
            x_in = self.align(x)[:, :, self.kt - 1:, :]   # align does nothing as c_in == c_out
        # print("x_in.shape: ", x_in.shape)
        if self.act == "GLU":
            # print("x.shape (GLU): ", x.shape)  torch.Size([32, 64, 27, 200])
            x_conv = self.conv(x)
            # print("x_conv.shape (GLU): ", x_conv.shape)  torch.Size([32, 128, 1, 200])
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            x_conv = self.conv(x)
            # print("x_conv.shape: ", x_conv.shape)
            return torch.sigmoid(x_conv + x_in)  
        return torch.relu(self.conv(x) + x_in)  


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks)) # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        ## [bs, channels, timesteps, nodes]
        # x.shape:  torch.Size([32, 32, 17, 128]) Lk.shape:  torch.Size([3, 128, 128])
        # x_c.shape :  torch.Size([32, 32, 17, 3, 128]) Lk.shape:  torch.Size([3, 128, 128])
        # x_gc.shape :  torch.Size([32, 32, 17, 128])
        # x_in.shape :  torch.Size([32, 32, 17, 128])
        # print("x.shape: ", x.shape, "Lk.shape: ", Lk.shape)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)              ## Ax      this is simply the multiplication of the adjacency matrix with the input for message passing. Each nodes updates as the sum of the nodes in its neighbourhood

        # print("x_c.shape : ", x_c.shape, "Lk.shape: ", Lk.shape)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b 
        # print("x_gc.shape : ", x_gc.shape)
        x_in = self.align(x) 
        # print("x_in.shape : ", x_in.shape)
        return torch.relu(x_gc + x_in)

class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = FCLayer(d_model, n_query) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2) # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        """
        :param x: key sequence of region embeding, nclv
        :return x: hidden embedding used for conv, ncqv
        :return x_agg: region embedding for spatial similarity, nvc
        :return A: temporal attention, lnv
        """
        # print("x.shape (before align): ", x.shape)
        x_in = self.align(x)[:, :, -self.n_query:, :] # ncqv
        # print("x_in.shape: ", x_in.shape)
        # calculate the attention matrix A using key x   
        # print("x.shape: ", x.shape, "self.att(x).shape: ", self.att(x).shape)
        A = self.att(x) # x: nclv, A: nqlv 
        A = F.softmax(A, dim=2) # nqlv
        # print("A.shape: ", A.shape)
        # calculate region embeding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2) # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg) # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        # print("A.shape: ", A.shape)
        A = self.softmax(self.agg(A).squeeze(2)) # A: lnqv->lnv
        # print("A.shape: ", A.shape)
        return torch.relu(x + x_in), x_agg.detach(), A.detach()

########################################
## An MLP predictor
########################################
class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        # print("x.shape before fc1: ", x.shape)   # torch.Size([32, 1, 200, 128])
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2))) # nlvc->nclv
        # print("x.shape after fc1: ", x.shape)
        x = self.fc2(x).permute(0, 2, 3, 1) # nclv->nlvc
        # print("x.shape after fc2: ", x.shape)
        return x

class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)  

    def forward(self, x):
        return self.linear(x)

class actuallyMLP(nn.Module):
    def __init__(self, c_in, c_out):
        super(actuallyMLP, self).__init__()
        self.fc1 = nn.Linear(c_in*200, int(c_in/2)*200)
        self.fc2 = nn.Linear(int(c_in/2)*200, c_out*200)
        self.c_out = c_out

    def forward(self, x):
        # print("x.shape: ", x.shape)   # torch.Size([32, 1, 200, 128])
        x = x.squeeze(1)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))             ##using a relu instead of a tanh as in the original MLP
        x = self.fc2(x)
        x = x.view(x.size(0), -1, self.c_out)
        x = x.unsqueeze(1)
        return x


class self_Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(self_Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        # self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)  ## [bs, seq_len, num_heads, dim_per_head]. transpose to [bs, num_heads, seq_len, dim_per_head]
        k = k.view(k.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        scores = F.softmax(scores, dim=-1)

        attended = torch.matmul(scores, v).transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)   ## do contigous to make sure the memory is contiguous. requirement of .view()

        output = self.out_linear(attended)
        # output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        return output
    
class cross_Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(cross_Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        # self.norm = nn.BatchNorm1d(d_model)


    def forward(self, reprA, reprB):
        q = self.q_linear(reprA)
        k = self.k_linear(reprB)
        v = self.v_linear(reprB)

        q = q.view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(k.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        scores = F.softmax(scores, dim=-1)

        attended = torch.matmul(scores, v).transpose(1, 2).contiguous().view(reprA.size(0), -1, self.d_model)

        output = self.out_linear(attended)
        # output = self.norm(attended.transpose(1, 2)).transpose(1, 2)
        return output
    
class PositionWise_cross_Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(PositionWise_cross_Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        # self.norm = nn.BatchNorm1d(d_model)


    def forward(self, reprA, reprB):
        q = self.q_linear(reprA)
        k = self.k_linear(reprB)
        v = self.v_linear(reprB)

        q = q.view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(k.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        scores = torch.sigmoid(scores)        ## make scores for each value bw 0-1 independent of others.
        scores = scores * torch.eye(scores.size(2)).to(scores.device)       ## zero all off diagonal elements of scores, to only update a1 with b1 and a2 with b2 and so on. 

        # scores.shape:  torch.Size([32, 4, 200, 200])
        # v.shape:  torch.Size([32, 4, 200, 16])
        # attended.shape:  torch.Size([32, 200, 64])
        # print("scores.shape: ", scores.shape)
        # print("v.shape: ", v.shape)
        attended = torch.matmul(scores, v).transpose(1, 2).contiguous().view(reprA.size(0), -1, self.d_model)
        # print("attended.shape: ", attended.shape)
        output = self.out_linear(attended)
        # output = self.norm(attended.transpose(1, 2)).transpose(1, 2)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        # output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        return output

        
"""
class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention = Attention(d_model, n_heads)

    def forward(self, x):

        x_og = x
        # x.shape: [bs, c, t, num_nodes]
        #print("x.shape: ", x.shape)
        # x = x.reshape(x.size(0), -1, x.size(3)).transpose(-2, -1)  # [bs, c*t, num_nodes] => [bs, num_nodes, c*t]   d_model = c*t
        #print("x.shape (after reshape): ", x.shape)
        # Apply attention
        x_skip = x
        x = self.attention(x)
        #print("x.shape (after attention): ", x.shape)
        x+=x_skip
        x_skip = x
        x = self.attention(x)
        x+=x_skip
        # Reshape x back to original shape
        x = x.transpose(-2, -1).reshape(x_og.shape)  # [bs, c, t, num_nodes]
        #print("x_out.shape: ", x.shape)
        return x    
"""

"""
class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        # self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # print("(input to spatial attention) x.shape: ", x.shape)   ## x.shape:  torch.Size([32, 32, 33, 200]) [b, c, t, n]
        ## lets call each head a timestep.
        x = x.permute(0, 3, 2, 1)  ## x.shape:  torch.Size([32, 200, 33, 32]) [b, n, t, c]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.transpose(1, 2)  ## q.shape:  torch.Size([32, 33, 200, 32]) [b, t, n, c]
        k = k.transpose(1, 2)  ## k.shape:  torch.Size([32, 33, 200, 32]) [b, t, n, c]
        v = v.transpose(1, 2)  ## v.shape:  torch.Size([32, 33, 200, 32]) [b, t, n, c]

        # q = q.view(q.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)  ## [bs, seq_len, num_heads, dim_per_head]. transpose to [bs, num_heads, seq_len, dim_per_head]
        # k = k.view(k.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        # v = v.view(v.size(0), -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        ## correct the normalizer 

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        scores = F.softmax(scores, dim=-1)

        attended = torch.matmul(scores, v).transpose(1, 2)   # .contiguous().view(x.size(0), -1, self.d_model)   ## do contigous to make sure the memory is contiguous. requirement of .view()
        # print("(output of spatial attention) attended.shape: ", attended.shape)   ## attended.shape:  torch.Size([32, 200, 33, 32])
        output = self.out_linear(attended)
        output = output.transpose(1, 3)
        # print("(output of spatial attention) x.shape: ", x.shape)     ## torch.Size([32, 200, 33, 32])
        # output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        return output
"""

""" looped implementation
class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_timesteps, n_heads):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_timesteps = n_timesteps

        # Time-dependent projection matrices
        self.q_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_timesteps)]) 
        self.k_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_timesteps)])
        self.v_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_timesteps)])

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  

        # Iterate over time steps
        outputs = []  # Store results from each time step
        for t in range(x.size(2)):  # Iterate over the 't' dimension
            q = self.q_linears[t](x[:, :, t, :]).unsqueeze(2) 
            k = self.k_linears[t](x[:, :, t, :]).unsqueeze(2)
            v = self.v_linears[t](x[:, :, t, :]).unsqueeze(2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
            scores = F.softmax(scores, dim=-1)

            attended = torch.matmul(scores, v).transpose(1, 2)   
            outputs.append(attended)

        # Concatenate outputs from all time steps
        output = torch.cat(outputs, dim=2)  

        output = self.out_linear(output)
        output = output.transpose(1, 3)
        return output
    """


class SpatialAttention(nn.Module):
    def __init__(self, d_model, n_timesteps, n_heads):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_timesteps = n_timesteps

        # Time-dependent projection matrices
        # self.q_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_timesteps)]) 
        self.q_linears = nn.parameter.Parameter(torch.randn(n_timesteps, d_model, d_model), requires_grad=True) 
        self.k_linears = nn.parameter.Parameter(torch.randn(n_timesteps, d_model, d_model), requires_grad=True) 
        self.v_linears = nn.parameter.Parameter(torch.randn(n_timesteps, d_model, d_model), requires_grad=True) 
        self.out_linears = nn.parameter.Parameter(torch.randn(n_timesteps, d_model, d_model), requires_grad=True) 

        # self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  

        # Iterate over time steps
        # print("x.size(): ", x.size())
        # print("q_linears.size(): ", self.q_linears.size())
        q = torch.einsum('bntc,tcd->bntd', x, self.q_linears)  # (batch_size, n_heads, n_timesteps, d_model)
        # print("q.size() after: ", q.size())
        k = torch.einsum('bntc,tcd->bntd', x, self.k_linears)  # (batch_size, n_heads, n_timesteps, d_model)
        v = torch.einsum('bntc,tcd->bntd', x, self.v_linears)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # print("q.size(): ", q.size())
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
        scores = F.softmax(scores, dim=-1)
        # print("scores.size(): ", scores.size())
        attended = torch.matmul(scores, v).transpose(1, 2)   
        # print("attended.size(): ", attended.size())

        output = torch.einsum('bntc,tcd->bntd', attended, self.out_linears)
        # output = self.out_linear(attended)
        # print("output.size(): ", output.size())
        output = output.transpose(1, 3)
        
        return output