#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import pandas as pd

# 검증 데이터 불러오기
test_query_data = pd.read_csv('./test_query_task2.csv', delimiter=',')

val_row_start = 0
val_row_end = 400000

test_data = pd.DataFrame({
    'customer': test_query_data.iloc[val_row_start:val_row_end, 2].values,
    'product': test_query_data.iloc[val_row_start:val_row_end, 1].values,
    #'label': gen_data.iloc[val_row_start:val_row_end, 6].values,
    'group': test_query_data.iloc[val_row_start:val_row_end, 5].values,
    'color': test_query_data.iloc[val_row_start:val_row_end, 3].values,
    'size': test_query_data.iloc[val_row_start:val_row_end, 4].values
})


# In[15]:


import torch
from torch_geometric.data import HeteroData

# 파일에서 HeteroData 객체 로드
loaded_hetero_data = torch.load('hetero_data.pth')


# In[12]:


gen_data = pd.read_csv('./train_valid_drop_indexing.csv', delimiter=',')


row = 30000000
data = pd.DataFrame({
    'order': gen_data.iloc[:row, 0].values,
    'customer': gen_data.iloc[:row, 2].values,
    'product': gen_data.iloc[:row, 1].values,
    'label': gen_data.iloc[:row, 6].values,
    'group': gen_data.iloc[:row, 5].values,
    'color': gen_data.iloc[:row, 3].values,
    'size': gen_data.iloc[:row, 4].values
})

# order, product, customer, color, size, group, label, indexing

data= data[:3000000]
# 고객과 제품 ID를 0부터 연속적으로 재매핑


unique_customers = data['customer'].unique()
unique_products = data['product'].unique()
unique_colors = data['color'].unique()
unique_sizes = data['size'].unique()
unique_groups = data['group'].unique()
num_customers = len(unique_customers)
num_products = len(unique_products)
num_colors =len(unique_colors)
num_sizes =len(unique_sizes)
num_groups =len(unique_groups)

customer_map = {id: idx for idx, id in enumerate(unique_customers)}
product_map = {id: idx for idx, id in enumerate(unique_products)}
color_map = {id: idx for idx, id in enumerate(unique_colors)}
size_map = {id: idx for idx, id in enumerate(unique_sizes)}
group_map = {id: idx for idx, id in enumerate(unique_groups)}


# In[13]:



def get_mapped_id(id, id_map, default_value=1):
    return id_map.get(id, default_value)
def preprocess_new_data(data):
    hetero_data_test = HeteroData()

    unique_customers = test_data['customer'].unique()
    unique_products = test_data['product'].unique()
    unique_colors = test_data['color'].unique()
    unique_sizes = test_data['size'].unique()
    unique_groups = test_data['group'].unique()
    

    hetero_data_test['customer'].x = torch.tensor([get_mapped_id(id, customer_map) for id in unique_customers]).view(-1, 1).float()
    hetero_data_test['product'].x = torch.tensor([get_mapped_id(id, product_map) for id in unique_products]).view(-1, 1).float()
    hetero_data_test['color'].x = torch.tensor([get_mapped_id(id, color_map) for id in unique_colors]).view(-1, 1).float()
    hetero_data_test['size'].x = torch.tensor([get_mapped_id(id, size_map) for id in unique_sizes]).view(-1, 1).float()
    hetero_data_test['group'].x = torch.tensor([get_mapped_id(id, group_map) for id in unique_groups]).view(-1, 1).float()
    
    edges = data[['customer', 'product']].apply(
    lambda row: [get_mapped_id(row['customer'], customer_map), get_mapped_id(row['product'], product_map)], axis=1)
    
    combined_edges_tensor = torch.tensor(edges.tolist(), dtype=torch.long).t()
    size_tensor = torch.tensor(data['size'].values, dtype=torch.long).view(-1, 1)
    color_tensor = torch.tensor(data['color'].values, dtype=torch.long).view(-1, 1)
    group_tensor = torch.tensor(data['group'].values, dtype=torch.long).view(-1, 1)
    # 엣지 속성 결합
    edge_attr_tensor = torch.cat([size_tensor, color_tensor, group_tensor], dim=1)


    # 엣지 데이터 추가
    hetero_data_test['customer', 'returns', 'product'].edge_index = combined_edges_tensor
    hetero_data_test['customer', 'returns', 'product'].edge_attr = edge_attr_tensor


    # 양방향 엣지 추가
    reversed_edges = combined_edges_tensor.flip(0)
    hetero_data_test['product', 'returned_by', 'customer'].edge_index = reversed_edges
    hetero_data_test['product', 'returned_by', 'customer'].edge_attr = edge_attr_tensor


    return hetero_data_test

# 새로운 데이터 전처리
new_hetero_data = preprocess_new_data(test_data)
new_hetero_data


# In[24]:


# 검증
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm 
 
import torch_geometric.transforms as T
from torch_geometric.nn import HANConv, Linear
 
def get_mapped_id(id, id_map, default_value=1):
    return id_map.get(id, default_value)

class HAN(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_channels=256, heads=8):
        super(HAN, self).__init__()
        self.han_conv = HANConv(dim_in, hidden_channels, heads=heads, dropout=0.2, metadata=loaded_hetero_data.metadata())
        
        self.color_dim = 8  # 임베딩 차원 조정
        self.size_dim = 4   # 임베딩 차원 조정
        self.group_dim = 4    # 임베딩 차원 조정
        
        self.color_embedding = nn.Embedding(num_colors, self.color_dim)
        self.size_embedding = nn.Embedding(num_sizes, self.size_dim)
        self.group_embedding = nn.Embedding(num_groups, self.group_dim)
        
        
        self.customer_embedding = nn.Embedding(num_customers, 16)
        self.product_embedding = nn.Embedding(num_products, 16)
        
        self.fc1 = nn.Linear(hidden_channels * 2 + self.color_dim + self.size_dim+ self.group_dim, 128)
        #self.fc1 = nn.Linear(hidden_channels * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(32)
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        customer_ids = torch.tensor([customer_map[cid] for cid in unique_customers], dtype=torch.long).to(device)
        product_ids = torch.tensor([product_map[cid] for cid in unique_products], dtype=torch.long).to(device)
        x_dict = {
            'customer': self.customer_embedding(customer_ids),
            'product': self.product_embedding(product_ids),
            'color': x_dict['color'],  # 원래 x_dict에서 color 추가
            'size': x_dict['size'],     # 원래 x_dict에서 size 추가
            'group': x_dict['group']     # 원래 x_dict에서 size 추가
            
        }
        #x_dict['customer'].x= self.customer_embedding(customer_ids)    
        color_embed = self.color_embedding(x_dict['color'].long())
        size_embed = self.size_embedding(x_dict['size'].long())
        group_embed = self.group_embedding(x_dict['group'].long())
        x_dict = self.han_conv(x_dict, edge_index_dict)
    
        
        edge_features_list = []
        edge_attr = new_hetero_data['customer', 'returns', 'product'].edge_attr
        #for i in range(edge_label_index.size(1)):
        for i in tqdm(range(edge_label_index.size(1)), desc="Processing edges"):
            src = edge_label_index[0, i]
            dst = edge_label_index[1, i]
            
            size_idx = edge_attr[i, 0].item()
            color_idx = edge_attr[i, 1].item()
            group_idx = edge_attr[i, 2].item()
            size_feature = size_embed[get_mapped_id(id, size_map)].view(-1)
            color_feature = color_embed[get_mapped_id(id, color_map)].view(-1)
            group_feature = group_embed[get_mapped_id(id, group_map)].view(-1)
            
            #size_feature = size_embed[size_map[size_idx]].view(-1)
            #color_feature = color_embed[color_map[color_idx]].view(-1)
            #group_feature = group_embed[group_map[group_idx]].view(-1)
        
            edge_feature = torch.cat([x_dict['customer'][src], x_dict['product'][dst], size_feature, color_feature, group_feature], dim=-1)
            #edge_feature = torch.cat([x_dict['customer'][src], x_dict['product'][dst]], dim=-1)
            edge_features_list.append(edge_feature)
        
        edge_features = torch.stack(edge_features_list, dim=0)
        
        x = F.relu(self.fc1(edge_features))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        return self.fc3(x)

best_model_path = 'HAN_best_model_.pth' #HAN_best_model_
#model = HeteroEmbeddingModel(num_customers, num_products, num_colors, num_sizes, embedding_dim)
model = HAN(dim_in=-1, dim_out=2)


model.load_state_dict(torch.load(best_model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

new_hetero_data, model = new_hetero_data.to(device), model.to(device)

model.eval()
#criterion = nn.BCELoss()
with torch.no_grad():
    out = model(new_hetero_data.x_dict, new_hetero_data.edge_index_dict, new_hetero_data['customer', 'returns', 'product'].edge_index)
    #edge_label = hetero_data['customer', 'returns', 'product'].edge_attr[:,0]
    preds = out.argmax(dim=1)
    print(preds )


# In[27]:


print(preds.sum() )


# In[ ]:


import os

GPU_NUM = 4


# 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


# In[ ]:




