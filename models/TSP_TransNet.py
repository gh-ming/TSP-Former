import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import numpy as np
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., return_att=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.return_att = return_att

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        if self.return_att:
            weights = attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x
        if self.return_att:
            return out, weights
        else:
            return out
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PhenoAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model=128, num_bases=12, temp_scale=30.):
        super().__init__()
        # 可学习物候基函数
        self.centers = nn.Parameter(torch.linspace(0, 365, num_bases))
        self.sigma = nn.Parameter(torch.ones(num_bases)*temp_scale)
        self.proj = nn.Linear(num_bases, d_model)
        
        # 季节相位编码
        self.season_emb = nn.Embedding(4, d_model)  # 春夏秋冬
        
    def get_season(self, doy):
        # 简单季节划分
        return torch.where(doy < 35, 0, # 冬
                          torch.where(doy < 126, 1, # 春
                                     torch.where(doy < 248, 2, # 夏
                                                 torch.where(doy < 339, 3, 0))))  # 秋 冬

    def forward(self, doy):
        """
        doy: [B,T] 实际日期数值（非one-hot）
        """
        # 高斯基函数编码
        basis = torch.exp(-(doy.unsqueeze(-1) - self.centers)**2 / 
                        (2*(self.sigma**2)))  # [B,T,K]
        time_emb = self.proj(basis)  # [B,T,d_model]
        
        # 季节相位编码
        season_idx = self.get_season(doy)  # [B,T]
        season_emb = self.season_emb(season_idx)  # [B,T,d_model]
        
        return time_emb + 0.3*season_emb

class PixelSetAggregationEncoder(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super(PixelSetAggregationEncoder, self).__init__()
        self.mlp = nn.Sequential( 
                                nn.Linear(in_features = in_channels,
                                        out_features=32),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features =32,
                                        out_features=64),
                                nn.ReLU(inplace=True) ,
                                )
        
        self.query_mlp = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=16),
                                        nn.ReLU(inplace=True))
        self.key_mlp  = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=16),
                                        nn.ReLU(inplace=True))
        self.value_mlp = nn.Sequential(nn.Linear(in_features =64,
                                            out_features=64),
                                        nn.ReLU(inplace=True))
    
    def forward(self,x):
        import ipdb;ipdb.set_trace()
        b,t,c,h,w= x.shape 
        x = x.permute(0,3,4,1,2).reshape(b*h*w*t,c)
        x = self.mlp(x)
        
        query = self.query_mlp(x.reshape(b,h,w,t,64)[:,2,2,:,:]).reshape(b,t,1,16) # the spectral features of central pixel
        key = self.key_mlp(x).reshape(b,h*w,t,16).permute(0,2,3,1)
        value = self.value_mlp(x).reshape(b,h*w,t,64).permute(0,2,1,3)
        
        scores = torch.matmul(query,key)
        weights = torch.softmax(scores,-1,dtype=torch.float16)
        
        x = x.reshape(b,h,w,t,64)[:,2,2,:,:] + torch.matmul(weights,value).squeeze(2)
        return x,weights
     
class TSP_TransNet(nn.Module):
    def __init__(self,num_classes=2,patch_size=3, d_model=128, n_head=4, depth=1, d_inner=64,dropout=0.2,scale_dim=4):
        super().__init__()
        self.modelname = self._get_name()
        self.num_classes = num_classes
        self.patch_size = patch_size 
        self.dim = d_model
        self.depth = depth
        self.heads = n_head
        self.dim_head = d_inner
        self.dropout = dropout
        self.emb_dropout = dropout
        self.scale_dim = scale_dim
        self.psae = PixelSetAggregationEncoder(in_channels=10)
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(10 *patch_size**2, self.dim),) #chw -> dim
        # self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.pheno_encoder = PhenoAwarePositionalEncoding(d_model=self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.depth + 2, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.temporal_decoder = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2),       
            nn.Unflatten(-1, (self.patch_size, self.patch_size))  # [B*T, H, W]
        )
        self.space_pos_embedding = nn.Parameter(torch.randn(1,self.dim))
        self.space_token = nn.Parameter(torch.randn(1,self.dim))
        self.space_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.tsp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, doy):
        import ipdb;ipdb.set_trace()
        B, T, C, H, W = x.shape
        center_h = H // 2
        center_w = W // 2
        x,weights = self.psae(x)
        x = x.reshape(B, T, C * H * W) # [B, T, 250]
        x = self.to_patch_embedding(x) # [B, T, D]

        # temporal embedding
        # xt = doy[:,:, 0, 0].to(torch.int64)  # [B, T]
        # xt = F.one_hot(xt, num_classes=365).to(torch.float32) # [B, T, 365]
        # xt = xt.reshape(-1, 365) # [B*T, 365]
        # temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim) # [B, T, D]
        xt = doy[:,:,0,0].to(torch.int64)  # [B,T] 直接使用原始DOY数值
        temporal_pos_embedding = self.pheno_encoder(xt)  # [B,T,d_model]

        x += temporal_pos_embedding # [B, T, D]
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B)  # [B, K, D]
        x = torch.cat((cls_temporal_tokens, x), dim=1) # [B, T+K, D]
        x = self.temporal_transformer(x)
        # NDVI alignment
        ndvi_pred = self.temporal_decoder(x[:, self.num_classes:].reshape(-1,self.dim))  # [B, T, D] -> [B,T,H*W]
        ndvi_pred = ndvi_pred.reshape(B,T,self.patch_size,self.patch_size) #[B, T, H, W]
        # spatial embedding
        x = x[:, :self.num_classes]  # [B, K, D]
        x = x.reshape(B * self.num_classes, self.dim) # [B*K, D]      
        x += self.space_pos_embedding
        x = self.dropout(x)
        x = x.reshape(B, self.num_classes, self.dim) # [B, K, D]
        x = self.space_transformer(x) # [B, K, D]
        # TSP alignment
        # import ipdb;ipdb.set_trace()
        tsp_pred = self.tsp_head(x[:,-1,:].reshape(-1, self.dim))
        tsp_pred = tsp_pred.reshape(B,H,W)
        # center_pred 
        x = self.mlp_head(x.reshape(-1, self.dim)) # [B*K, D] -> [B*K, H*W]
        x = x.reshape(B, self.num_classes,H,W)  # [B, K, H, W] 
        x = x[:,:,center_h,center_w] # [B, K]
        return x,ndvi_pred,tsp_pred

if __name__ == '__main__':
    x = torch.rand((64, 12, 10, 5, 5))
    doy = torch.rand((64, 12,5,5))
    weight = torch.rand((64, 12,5,5))

    # model = TViT(model_config).cuda()
    # model = STViT(model_config)#.cuda()
    model = TSP_TransNet(num_classes=2, d_model=128, n_head=4, depth=1, d_inner=64, dropout=0.2, scale_dim=4)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model((x, doy, weight))

    # torch.norm(cls, dim=2).shape
    print("Shape of out :", out.shape)  # [B, num_classes]