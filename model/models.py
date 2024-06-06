import torch.nn as nn
import torch
# import 
from lib.utils import masked_mae_loss
from model.aug import (
    aug_topology, 
    aug_traffic, 
)
from model.layers import (
    STEncoder, 
    SpatialHeteroModel, 
    TemporalHeteroModel, 
    MLP,
    actuallyMLP,
    self_Attention,
    cross_Attention, 
    PositionWise_cross_Attention,
    PositionwiseFeedForward,
)
import numpy as np
class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        # spatial temporal encoder
        
        # self.encoderC = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        # self.encoderD = STEncoder(Kt=3, Ks=3, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
        #                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        
        # self.channel_reducer1 = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 1, 1), padding='same') ## padding='same' to keep output size same as input 
        # self.channel_reducer2 = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 1, 1), padding='same') ## padding='same' to keep output size same as input 
        # self.channel_reducer = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 1, 1), padding='same') ## padding='same' to keep output size same as input 

        self.attention1 = self_Attention(128, 4)
        self.attention2 = self_Attention(128, 4)
        # self.attentionA1 = self_Attention(64, 4)
        # self.attentionA2 = self_Attention(64, 4)
        # self.attentionB1 = self_Attention(64, 4)
        # self.attentionB2 = self_Attention(64, 4)
        # self.add_attentionA1 = self_Attention(64, 4)
        # self.add_attentionB1 = self_Attention(64, 4)
        # self.add_attentionA2 = self_Attention(64, 4)
        # self.add_attentionB2 = self_Attention(64, 4)


        # self.cross_attention1 = PositionWise_cross_Attention(64, 4)
        # self.cross_attention2 = PositionWise_cross_Attention(64, 4)

        self.ff = PositionwiseFeedForward(d_model=128, d_ff=64*4)
        # self.ffA1 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # self.ffB1 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # self.ffA2 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # self.ffB2 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # self.ffCA1 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # self.ffCA2 = PositionwiseFeedForward(d_model=64, d_ff=64*4)
        # traffic flow prediction branch
        self.mlp = MLP(2*args.d_model, args.d_output)
        # self.mlpRepr = MLP(2*args.d_model, args.d_model)
        # temporal heterogenrity modeling branch
        # self.thm = TemporalHeteroModel(args.d_model, args.batch_size, args.num_nodes, args.device)
        # spatial heterogenrity modeling branch
        # self.shm = SpatialHeteroModel(args.d_model, args.nmb_prototype, args.batch_size, args.shm_temp)
        self.mae = masked_mae_loss(mask_value=5.0)
        # self.mae = masked_mae_loss(mask_value=None)
        self.args = args
        # adj = args.graph_file
        # adj = np.load(adj)["adj_mx"]
        graph_init = args.graph_init
        
        ## attention flags
        self.self_attention_flag = args.self_attention_flag
        self.cross_attention_flag = args.cross_attention_flag
        self.feedforward_flag = args.feedforward_flag
        self.layer_norm_flag = args.layer_norm_flag
        self.additional_sa_flag = args.additional_sa_flag

        if graph_init == "eye":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.eye(adj.shape[1]).float(), requires_grad=False)
        elif graph_init == "zeros":             ## eye sconv because of cheb approximation. actual adj will be: [eye, zero, zero]
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.zeros_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "no_sconv":             ## no sconv because of sconv flag, but still need to set some placeholder value to run.
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.zeros_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "ones":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.ones_like(torch.tensor(adj).float()), requires_grad=False)
        elif graph_init == "random":
            adj = "data/NYCTaxi/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.empty_like(torch.tensor(adj).float()), requires_grad=False)
            self.xavier_uniform_init(self.learnable_graph)

        elif graph_init == "low_rank" and args.learnable_flag == True:
            r = int(args.num_nodes * (2/3))
            # r = int(args.rank)
            num_low_rank_matrices = 3

            self.matrices1 = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, r), requires_grad=True).to(args.device)
            self.matrices2 = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, r), requires_grad=True).to(args.device)
            # self.learnable_graph = nn.Parameter(torch.randn(num_low_rank_matrices, args.num_nodes, args.num_nodes), requires_grad=True).to(args.device)
            # self.learnable_graph = torch.matmul(self.matrices1, self.matrices2.transpose(1, 2))


        elif graph_init == "8_neighbours" and args.learnable_flag == True:
            adj = np.expand_dims(adj, axis=0)  
            adj = np.repeat(adj, 3, axis=0)
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=True)
        
        elif graph_init == "eye":
            adj = f"data/{args.dataset}/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.eye(adj.shape[1]).float(), requires_grad=False)
        
        elif graph_init == "8_neighbours" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_mx.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)
        
        elif graph_init == "shared_lpe_T" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_lpe_shared_T.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        elif graph_init == "shared_lpe_T_T" and args.learnable_flag == False:
            adj = f"data/{args.dataset}/adj_lpe_shared_T_T.npz"
            adj = np.load(adj)["adj_mx"]
            self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

        # elif graph_init == "pre_trained_symmetric" and args.learnable_flag == False:
        #     adj = "data/NYCTaxi/V2maskedAttentionADJ_S.npz"
        #     adj = np.load(adj)["adj_mx"]
        #     self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)


        self.encoderA = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag)
        self.encoderB = STEncoder(Kt=3, Ks=args.cheb_order, blocks=[[2, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout, graph_init=graph_init, learnable_flag=args.learnable_flag)         
        # self.learnable_graph = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=True)
        # self.learnable_graph = nn.Parameter(torch.zeros_like(torch.tensor(adj).float()), requires_grad=True)
        # self.learnable_graph = nn.Parameter(torch.eye(adj.shape[1]).float(), requires_grad=False)

        # nn.init.xavier_uniform_(self.learnable_graph)        
    
        ## norms
        self.layernorm1 = nn.LayerNorm(128)
        self.layernorm2 = nn.LayerNorm(128)
        self.layernorm3 = nn.LayerNorm(128)
        # self.layernorm4 = nn.LayerNorm(64)
        # self.layernorm5 = nn.LayerNorm(64)
        # self.layernorm6 = nn.LayerNorm(64)
        # self.layernorm7 = nn.LayerNorm(64)
        # self.layernorm8 = nn.LayerNorm(64)
        # self.layernorm9 = nn.LayerNorm(64)
        # self.layernorm10 = nn.LayerNorm(64)
        # self.layernorm11 = nn.LayerNorm(64)
        # self.layernorm12 = nn.LayerNorm(64)
        # self.layernorm13 = nn.LayerNorm(64)
        # self.layernorm14 = nn.LayerNorm(64)
        # self.layernorm15 = nn.LayerNorm(64)
        # self.layernorm16 = nn.LayerNorm(64)

        self.dataset = args.dataset

    def xavier_uniform_init(self, tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        nn.init.uniform_(tensor, -std, std) 

    def forward(self, view1, graph):
        # input_sequence_dict = {"A":[-4, 19], "B":[-9, -4], "C":[-14, -9], "D":[-19, -14]}
        # input_sequence_dict = {"A":[-8, 35], "B":[-17, -8], "C":[-26, -17], "D":[-35, -26]}
        # print("view1.shape: ", view1.shape, "graph.shape: ", graph.shape)  # view1.shape:  torch.Size([32, 19, 128, 2]) graph.shape:  torch.Size([128, 128])
        
        # view1A = view1[:, -8:35, :, :]
        # view1B = view1[:, -17:-8, :, :]
        # view1C = view1[:, -26:-17, :, :]
        # view1D = view1[:, -35:-26, :, :]
        if self.dataset == "NYCBike1":
            view1A = view1[:, -4:19, :, :]
            view1B = view1[:, -9:-4, :, :]
        elif self.dataset == "NYCBike2" or self.dataset == "NYCTaxi" or self.dataset == "BJTaxi": 
            view1A = view1[:, -8:35, :, :]
            view1B = view1[:, -17:-8, :, :]
        # view1C = view1[:, -14:-9, :, :]
        # view1D = view1[:, -19:-14, :, :]
        # print("\n\n graph.shape: ", graph.shape)  ## graph.shape:  torch.Size([128, 128])

        # print("view1A.shape: ", view1A.shape, "view1B1.shape: ", view1B1.shape, "view1B2.shape: ", view1B2.shape, "view1B3.shape: ", view1B3.shape)
        # view1B = torch.cat((view1B1, view1B2, view1B3), dim=1)
        # print("\n\nview1B.shape: ", view1B.shape)  ## view1B.shape:  torch.Size([32, 3, 5, 128, 2])
        # view1BIN = view1B[..., 0].unsqueeze(-1)
        # print("view1BIN.shape: ", view1BIN.shape)  ## view1BIN.shape:  torch.Size([32, 3, 5, 128])  unsqueeze(-1) to get last dim back
        # view1BOUT = view1B[..., 1].unsqueeze(-1)
        
        # view1B = self.channel_reducer(view1B).squeeze(1)
        
        
        # view1BIN = self.channel_reducer1(view1BIN).squeeze(1)
        # view1BOUT = self.channel_reducer2(view1BOUT).squeeze(1)
        # view1B = torch.cat((view1BIN, view1BOUT), dim=-1)
        # print("view1B_reduced.shape: ", view1B_reduced.shape)
        
        # learnable_graph = torch.matmul(self.matrices1, self.matrices2.transpose(1, 2))
        # learnable_graph = torch.relu(learnable_graph)

        ## softmax with temp
        # T=10
        # learnable_graph = learnable_graph / T
        # learnable_graph = torch.softmax(learnable_graph, dim=1)
        learnable_graph = self.learnable_graph
        # print("learnable_graph.shape: ", learnable_graph.shape)  
        # print(f"type(learnable_graph): {type(learnable_graph)}")
        
        """
        # import matplotlib.pyplot as plt
        # plt.matshow(learnable_graph.cpu().detach().numpy())
        # plt.show()
        """
        repr1A = self.encoderA(view1A, learnable_graph) # view1: n,l,v,c; graph: v,v 
        repr1B = self.encoderB(view1B, learnable_graph) # view1: n,l,v,c; graph: v,v 
        # repr1C = self.encoderC(view1C, graph) # view1: n,l,v,c; graph: v,v 
        # repr1D = self.encoderD(view1D, graph) # view1: n,l,v,c; graph: v,v 
        
        # print("repr1A.shape: ", repr1A.shape) # repr1A.shape:  torch.Size([32, 1, 128, 64])
        # print("repr1B.shape: ", repr1B.shape) # repr1B.shape:  torch.Size([32, 1, 128, 64])
        
        #### combine the representation from EncoderA and EncoderB ####
        # before cross attention, first lets update A and B using self attention
        
        combined_repr = torch.cat((repr1A, repr1B), dim=3)            ## combine along the channel dimension d_model
        # combined_repr = self.mlpRepr(combined_repr)
        if self.self_attention_flag:
            combined_repr = combined_repr.squeeze(1)
            if self.feedforward_flag:
                combined_repr_copy = combined_repr
                combined_repr = self.ff(combined_repr)
                combined_repr = combined_repr + combined_repr_copy
                if self.layer_norm_flag == True:
                    combined_repr = self.layernorm1(combined_repr)
            # combined_repr_copy = combined_repr
            # combined_repr = self.ffA1(combined_repr)   ## for inter channel mixing
            # combined_repr = combined_repr + combined_repr_copy
            çombined_repr_copy = combined_repr
            combined_repr = self.attention1(combined_repr)
            combined_repr = combined_repr + çombined_repr_copy  # skip connection
            if self.layer_norm_flag == True:
                combined_repr = self.layernorm2(combined_repr)
            combined_repr_copy = combined_repr
            combined_repr = self.attention2(combined_repr)
            combined_repr = combined_repr + combined_repr_copy  # skip connection
            if self.layer_norm_flag == True:
                combined_repr = self.layernorm3(combined_repr)
            combined_repr = combined_repr.unsqueeze(1)
        


        """
        print("combined_repr.shape: ", combined_repr.shape)
        if self.self_attention_flag == True:
            repr1A = repr1A.squeeze(1)
            repr1B = repr1B.squeeze(1)

            repr1A_copy = repr1A
            repr1B_copy = repr1B

            repr1A = self.attentionA1(repr1A)
            repr1B = self.attentionB1(repr1B)  
            
            repr1A = repr1A + repr1A_copy  # skip connection
            repr1B = repr1B + repr1B_copy  # skip connection

            if self.layer_norm_flag == True:
                repr1A = self.layernorm1(repr1A)
                repr1B = self.layernorm2(repr1B)

            repr1A_copy = repr1A
            repr1B_copy = repr1B

            if self.feedforward_flag == True:
                repr1A = self.ffA1(repr1A)
                repr1B = self.ffB1(repr1B)

                repr1A = repr1A + repr1A_copy  # skip connection
                repr1B = repr1B + repr1B_copy  # skip connection
                
                
                if self.layer_norm_flag == True:
                    repr1A = self.layernorm3(repr1A)
                    repr1B = self.layernorm4(repr1B)

                repr1A_copy = repr1A
                repr1B_copy = repr1B
            
            repr1A = self.attentionA2(repr1A)
            repr1B = self.attentionB2(repr1B)  
            
            repr1A = repr1A + repr1A_copy  # skip connection
            repr1B = repr1B + repr1B_copy  # skip connection
            
            if self.layer_norm_flag == True:
                repr1A = self.layernorm5(repr1A)
                repr1B = self.layernorm6(repr1B)

            repr1A_copy = repr1A
            repr1B_copy = repr1B
            
            if self.feedforward_flag == True:
                repr1A = self.ffA2(repr1A)
                repr1B = self.ffB2(repr1B)

                repr1A = repr1A + repr1A_copy  # skip connection
                repr1B = repr1B + repr1B_copy  # skip connection
                
                
                if self.layer_norm_flag == True:
                    repr1A = self.layernorm7(repr1A)
                    repr1B = self.layernorm8(repr1B)

            repr1A = repr1A.unsqueeze(1)
            repr1B = repr1B.unsqueeze(1)

        if self.cross_attention_flag == True:
            repr1A = repr1A.squeeze(1)
            repr1B = repr1B.squeeze(1)
            
            ### start: 1x cross attention

            repr1A_copy = repr1A 
            repr1A = self.cross_attention1(repr1A, repr1B)    ## need to update A using info from B and not the other way around. Cuz A is the most relevant info.      
            repr1A = repr1A + repr1A_copy  # skip connection
            
            if self.layer_norm_flag == True:
                repr1A = self.layernorm9(repr1A)
            repr1A_copy = repr1A
            ### end


            ### start: 1x ff
            if self.feedforward_flag == True:
                repr1A = self.ffCA1(repr1A)
                repr1A = repr1A + repr1A_copy  # skip connection
                
                if self.layer_norm_flag == True:
                    repr1A = self.layernorm10(repr1A)           
                repr1A_copy = repr1A
            ### end


            if self.additional_sa_flag == True:
                repr1A_copy = repr1A
                repr1B_copy = repr1B

                repr1A = self.add_attentionA1(repr1A)
                repr1B = self.add_attentionB1(repr1B)  
                
                repr1A = repr1A + repr1A_copy  # skip connection
                repr1B = repr1B + repr1B_copy  # skip connection

                if self.layer_norm_flag == True:
                    repr1A = self.layernorm13(repr1A)
                    repr1B = self.layernorm14(repr1B)
                
                repr1A_copy = repr1A
                repr1B_copy = repr1B
               
            repr1A = self.cross_attention2(repr1A, repr1B)    ## need to update A using info from B and not the other way around. Cuz A is the most relevant info.   
            repr1A = repr1A + repr1A_copy  # skip connection
            
            if self.layer_norm_flag == True:
                repr1A = self.layernorm11(repr1A)
            repr1A_copy = repr1A
            
            
            if self.feedforward_flag == True:
                repr1A = self.ffCA2(repr1A)
                repr1A = repr1A + repr1A_copy  # skip connection
                
                if self.layer_norm_flag == True:
                    repr1A = self.layernorm12(repr1A)
            
            
            if self.additional_sa_flag == True:
                repr1A_copy = repr1A
                repr1B_copy = repr1B

                repr1A = self.add_attentionA2(repr1A)
                repr1B = self.add_attentionB2(repr1B)  
                
                repr1A = repr1A + repr1A_copy  # skip connection
                repr1B = repr1B + repr1B_copy  # skip connection

                if self.layer_norm_flag == True:
                    repr1A = self.layernorm15(repr1A)
                    repr1B = self.layernorm16(repr1B)
                
                repr1A_copy = repr1A
                repr1B_copy = repr1B
               
            
            repr1A = repr1A.unsqueeze(1)

        """
        


        #### combine the representation from EncoderA and EncoderB ####
        
        ## now 2*d_model --> d_model
        # print("combined_repr.shape: ", combined_repr.shape)
        # combined_repr = self.mlpRepr(combined_repr)
        # print("combined_repr.shape: ", combined_repr.shape)
        # s_sim_mx = self.fetch_spatial_sim()
        # graph2 = aug_topology(s_sim_mx, graph, percent=self.args.percent*2)
        
        # t_sim_mx = self.fetch_temporal_sim()
        # view2 = aug_traffic(t_sim_mx, view1, percent=self.args.percent)
        # print("view2.shape: ", view2.shape, "graph2.shape: ", graph2.shape)
        # repr2 = self.encoder(view2, graph2)
        repr2 = None
        return combined_repr, learnable_graph

    def fetch_spatial_sim(self):
        """
        Fetch the region similarity matrix generated by region embedding.
        Note this can be called only when spatial_sim is True.
        :return sim_mx: tensor, similarity matrix, (v, v)
        """
        return self.encoder.s_sim_mx.cpu()
    
    def fetch_temporal_sim(self):
        return self.encoder.t_sim_mx.cpu()

    def predict(self, z1, z2):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        # print("z1.shape: ", z1.shape)
        return self.mlp(z1)

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        loss = loss_weights[0] * l1 

        # l2 = self.temporal_loss(z1, z2)
        # sep_loss.append(l2.item())
        # if self.args.T_Loss==1:
            # loss += loss_weights[1] * l2
        
        # l3 = self.spatial_loss(z1, z2)
        # sep_loss.append(l3.item())
        # if self.args.S_Loss==1:
            # print("spatial loss: ", l3)
            # loss += loss_weights[2] * l3 
        # print("predLoss: ", l1.item(), "temporalLoss: ", l2.item(), "spatialLoss: ", l3.item())
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(z1, z2))
        y_true = scaler.inverse_transform(y_true)
 
        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss
    
    def temporal_loss(self, z1, z2):
        return self.thm(z1, z2)

    def spatial_loss(self, z1, z2):
        return self.shm(z1, z2)
    