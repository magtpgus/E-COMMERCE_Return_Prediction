#!/usr/bin/env python
# coding: utf-8

# In[47]:


import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from torch_geometric.nn import HANConv
from torch import nn
import torch.nn.functional as F

"""

valid & train 데이터를 통합한 train_valid_drop_indexing.csv을 가져오고
validation에 해당하는 데이터를 마스킹하여 검증 진행

"""
gen_data = pd.read_csv('./train_valid_drop_indexing.csv', delimiter=',')
train_data = pd.read_csv('./train_indexing.csv', delimiter=',')
val_indexing_data = pd.read_csv('./valid_indexing.csv', delimiter=',')


row = 5000000
data = pd.DataFrame({
    'order': gen_data.iloc[:row, 0].values,
    'customer': gen_data.iloc[:row, 2].values,
    'product': gen_data.iloc[:row, 1].values,
    'label': gen_data.iloc[:row, 6].values,
    'group': gen_data.iloc[:row, 5].values,
    'color': gen_data.iloc[:row, 3].values,
    'size': gen_data.iloc[:row, 4].values
})

val_data = pd.DataFrame({
    'customer': val_indexing_data.iloc[:row, 2].values,
    'product': val_indexing_data.iloc[:row, 1].values,
    'label': val_indexing_data.iloc[:row, 6].values,
    'group': val_indexing_data.iloc[:row, 5].values,
    'color': val_indexing_data.iloc[:row, 3].values,
    'size': val_indexing_data.iloc[:row, 4].values
})


# order, product, customer, color, size, group, label, indexing

data = data.sample(frac=1).reset_index(drop=True)
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


# In[38]:


# validation 데이터들 마스킹 할당
file_path = 'task2_valid_label.txt'

with open(file_path, 'r') as file:
    file_content = file.read()
lines_valid_label = file_content.strip().split('\n')

mapping_df = pd.DataFrame([x.split('\t') for x in lines_valid_label], columns=['order', 'product', 'label'])

mapping_df['order'] = mapping_df['order'].astype(int)
mapping_df['product'] = mapping_df['product'].astype(int)
mapping_df['label'] = mapping_df['label'].astype(int)
merged_data = data.merge(mapping_df, on=['order', 'product', 'label'], how='left', indicator=True)

merged_data['not_in_list'] = merged_data['_merge'] == 'left_only'

result = merged_data['not_in_list']
train_mask = torch.tensor(result.values)
valid_mask = ~train_mask


# In[39]:


#cos_similar_customers.txt
sim_file = 'cos_similar_customers.txt'
import ast

cos_similar_customers = {}
with open(sim_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        customer_id, similarities = line.split(': ')
        customer_id = int(customer_id)
        similarities = ast.literal_eval(similarities.strip())
        cos_similar_customers[customer_id] = similarities
filtered_similarities= cos_similar_customers


# In[40]:


#매핑된 id 
customer_map = {id: idx for idx, id in enumerate(unique_customers)}
product_map = {id: idx for idx, id in enumerate(unique_products)}
color_map = {id: idx for idx, id in enumerate(unique_colors)}
size_map = {id: idx for idx, id in enumerate(unique_sizes)}
group_map = {id: idx for idx, id in enumerate(unique_groups)}

hetero_data = HeteroData()
hetero_data['customer'].x = torch.tensor([customer_map[id] for id in unique_customers]).view(-1, 1).float()
hetero_data['product'].x = torch.tensor([product_map[id] for id in unique_products]).view(-1, 1).float()
hetero_data['color'].x = torch.tensor([color_map[id] for id in unique_colors]).view(-1, 1).float()
hetero_data['size'].x = torch.tensor([size_map[id] for id in unique_sizes]).view(-1, 1).float()
hetero_data['group'].x = torch.tensor([group_map[id] for id in unique_groups]).view(-1, 1).float()


# In[41]:


#유사도 양수/음수 나눠서 다른 엣지로 저장 
sim_edges_positive = set()
sim_edges_negative = set()

for customer, similar_list in filtered_similarities.items():
    customer_idx = customer_map.get(int(customer))
    if customer_idx is None:
        continue
    for similar_customer in similar_list:
        similar_customer_idx = customer_map.get(abs(int(similar_customer))) #양수로 고객id 변환
        if similar_customer_idx is None:
            continue
        if similar_customer>0:
            edge = (customer_idx, similar_customer_idx)
            reversed_edge = (similar_customer_idx, customer_idx)
            # 양방향 엣지 추가
            sim_edges_positive.add(edge)
            sim_edges_positive.add(reversed_edge)
        else:
            edge = (customer_idx, abs(similar_customer_idx))
            reversed_edge = (abs(similar_customer_idx), customer_idx)
            # 양방향 엣지 추가
            sim_edges_negative.add(edge)
            sim_edges_negative.add(reversed_edge)


# In[42]:


sim_positive = list(sim_edges_positive)
sim_negative = list(sim_edges_negative)
sim_positive_tensor = torch.tensor(sim_positive, dtype=torch.long).t()
sim_negative_tensor = torch.tensor(sim_negative, dtype=torch.long).t()

hetero_data['customer', 'positive_to', 'customer'].edge_index = sim_positive_tensor


# In[43]:


edges = data[['customer', 'product']].apply(lambda row: [customer_map[row['customer']], product_map[row['product']]], axis=1)
combined_edges_tensor = torch.tensor(edges.tolist(), dtype=torch.long).t()

# 구매와 반품 라벨을 결합한 텐서 생성
edge_labels_tensor = torch.tensor(data['label'].values, dtype=torch.long)


# 사이즈, 색상, 그룹 정보를 엣지 속성으로 추가
size_tensor = torch.tensor(data['size'].values, dtype=torch.long).view(-1, 1)
color_tensor = torch.tensor(data['color'].values, dtype=torch.long).view(-1, 1)
group_tensor = torch.tensor(data['group'].values, dtype=torch.long).view(-1, 1)



# 엣지 속성 결합
#edge_attr_tensor = torch.cat([size_tensor, color_tensor, group_tensor,edge_labels_tensor.view(-1, 1)], dim=1)
edge_attr_tensor = torch.cat([size_tensor, color_tensor, group_tensor,edge_labels_tensor.view(-1, 1)], dim=1)

# 엣지 데이터 추가
hetero_data['customer', 'returns', 'product'].edge_index = combined_edges_tensor
hetero_data['customer', 'returns', 'product'].edge_attr = edge_attr_tensor
    

# 양방향 엣지 추가
reversed_edges = combined_edges_tensor.flip(0)
hetero_data['product', 'returned_by', 'customer'].edge_index = reversed_edges
#hetero_data['product', 'returned_by', 'customer'].edge_attr = edge_labels_tensor.view(-1, 1)
hetero_data['product', 'returned_by', 'customer'].edge_attr = edge_attr_tensor


#print(hetero_data, '\n','edges: ', edges,'\n','combined_edges_tensor: ', combined_edges_tensor,"reversed_edges",reversed_edges)
#edge_labels_tensor


# In[44]:


def apply_masks_to_bidirectional_edges(data, edge_type,reverse_edge_type):
    # validation을 위한 mask 적용
    data[edge_type].train_mask = train_mask
    data[edge_type].val_mask = valid_mask
    #data[edge_type].test_mask = test_mask
    
    data[reverse_edge_type].train_mask = train_mask
    data[reverse_edge_type].val_mask = valid_mask
    #data[reverse_edge_type].test_mask = test_mask
    

# 양방향 엣지에 동일한 마스크 적용
apply_masks_to_bidirectional_edges(hetero_data, ('customer', 'returns', 'product'), ('product', 'returned_by', 'customer'))


# In[45]:


import torch
import torch.nn.functional as F
from torch import nn
 
import torch_geometric.transforms as T
from torch_geometric.nn import HANConv, Linear
 
def get_mapped_id(id, id_map, default_value=1):
    return id_map.get(id, default_value)

class HAN(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_channels=256, heads=8):
        super(HAN, self).__init__()
        self.han_conv = HANConv(dim_in, hidden_channels, heads=heads, dropout=0.2, metadata=hetero_data.metadata())
        
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
        edge_attr = hetero_data['customer', 'returns', 'product'].edge_attr
        for i in range(edge_label_index.size(1)):
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


# In[46]:


from sklearn.metrics import recall_score
from tqdm import tqdm
model = HAN(dim_in=-1, dim_out=2)
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
hetero_data, model = hetero_data.to(device), model.to(device)
edge_labels_tensor= edge_labels_tensor.to(device)

# 훈련 및 평가 결과 저장을 위한 리스트 초기화
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_recalls = []



def train():
    model.train()
    optimizer.zero_grad()
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict,hetero_data['customer', 'returns', 'product'].edge_index)
    #print("out:",out)
    mask = hetero_data['customer', 'returns', 'product'].train_mask
    
    edge_label = hetero_data['customer', 'returns', 'product'].edge_attr[:,3]
    edge_label2 = edge_labels_tensor
    loss = F.cross_entropy(out[mask], edge_label2[mask])
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(mask):
    model.eval()
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict, hetero_data['customer', 'returns', 'product'].edge_index)
    edge_label = hetero_data['customer', 'returns', 'product'].edge_attr[:,3]
    edge_label2 = edge_labels_tensor
    
    preds = out.argmax(dim=1)
    acc = (preds[mask] == edge_label2[mask]).sum().item() / mask.sum()
    recall = recall_score(edge_label[mask].cpu(), preds[mask].cpu(), average='macro')
    loss = F.cross_entropy(out[mask], edge_label[mask]).cpu().item()
    
    return acc, recall, loss
# 모델 저장을 위한 설정
best_val_acc = 0.0
best_model_path = 'HAN_best_model_.pth'

for epoch in range(1, 35):
    loss = train()
    if epoch % 1 == 0:
        train_acc,_,k = test(hetero_data['customer', 'returns', 'product'].train_mask) #test()
        val_acc,recall,val_loss = test(hetero_data['customer', 'returns', 'product'].val_mask) #test()
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        train_losses.append(k)
        val_recalls.append(recall)
        
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f},Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_acc:.4f}, lr: {optimizer.param_groups[0]["lr"]}')
         
        #train_accuracies.append(train_acc)
    # 모델 저장
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with validation accuracy: {best_val_acc:.4f}')
print(f'Best val accuracy: {best_val_acc:.4f}')


# In[22]:



import matplotlib.pyplot as plt

train_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses]
val_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in val_losses]
train_acc_cpu = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accuracies]
val_acc_cpu = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in val_accuracies]


epochs = range(1, len(val_losses_cpu) + 1)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_cpu, label='Train Loss')
plt.plot(epochs, val_losses_cpu, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_cpu, label='Train Accuracy')
plt.plot(epochs, val_acc_cpu, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


import os

GPU_NUM = 2


# 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


# In[ ]:




