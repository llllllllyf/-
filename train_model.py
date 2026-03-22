# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from layer.DotaModelM1V128K10 import *
import pandas as pd
import numpy as np

import warnings,os,gc
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = 'data'
if not os.path.exists(path + "/embfeatures"):
    os.mkdir(path + "/embfeatures")
if not os.path.exists(path + "/id2index"):
    os.mkdir(path + "/id2index")

#================================================= Data =================================================
feed_info = pd.read_csv(path + '/feed_info.csv')
user_action = pd.read_csv(path + '/user_action.csv').drop_duplicates(subset=["userid", "feedid"], keep="last").reset_index(drop=True)
label_cols = ['read_comment','like','click_avatar','forward','favorite','comment','follow']+['play']
# 视频信息，用户信息
#================================================= ID2Index =================================================
# id映射器
def id_encode(series):
    unique = list(series.unique())
    unique.sort()
    return dict(zip(unique, range(series.nunique()))) #, dict(zip(range(series.nunique()), unique))

# 加载用户和视频 id映射文件，如果没有，构建一个并保存，后续使用直接加载就行了
if not os.path.exists(path + "/id2index/userid2index.npy"):
    userid2index = id_encode(user_action.userid)    # key: userid,  value: index
    feedid2index = id_encode(feed_info.feedid)
    authorid2index = id_encode(feed_info.authorid)

    np.save(path + "/id2index/userid2index.npy",userid2index)
    np.save(path + "/id2index/feedid2index.npy",feedid2index)
    np.save(path + "/id2index/authorid2index.npy",authorid2index)
else:
    userid2index = np.load(path + "/id2index/userid2index.npy",allow_pickle=True).item()
    feedid2index = np.load(path + "/id2index/feedid2index.npy",allow_pickle=True).item()
    authorid2index = np.load(path + "/id2index/authorid2index.npy",allow_pickle=True).item()

# 读取指定词和词向量，并保存为字典的形式  {word:embedding}
def read_id_dict(name, dim_n, emb_mode):
    p=path + f'/embfeatures/{name}_{emb_mode}_{dim_n}.pickle'
    tmp = pd.read_pickle(p)
    tmp_dict = {}
    for i,item in zip(tmp[tmp.columns[0]].values, tmp[tmp.columns[1:]].values):
        tmp_dict[i] = item
    return tmp_dict

# 读取词向量矩阵，将其储存为二维矩阵形式
def embedding_mat(feat_name, dim_n, emb_mode):
    #读取id的词向量以及id映射表
    model = read_id_dict(feat_name,dim_n,emb_mode)
    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")
    # 根据ID在模型中的嵌入向量填充矩阵，并多建立一个未知向量
    embed_matrix = np.zeros((len(id2index) + 1, dim_n))
    for word, i in id2index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            embed_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(dim_n) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embed_matrix[i] = unk_vec
    
    return embed_matrix

# 特征id转换为整数并为序列填充pad
def pad_seq(df, feat_name, max_len=1, mode='data'):
    tmp = df[[feat_name]].copy()

    if feat_name.startswith("user"):
        id2index = userid2index
    elif feat_name.startswith("feed"):
        id2index = feedid2index
    elif feat_name.startswith("author"):
        id2index = authorid2index
    else:
        print("Feat Name Error!!!")

    tmp[feat_name] = tmp[feat_name].apply(lambda x:id2index[x])
    seq = pad_sequences(tmp.values, maxlen = max_len)
    return seq
def calculate_auc(y_true, y_pred):
    """计算8个目标的AUC"""
    auc_scores = []
    target_names = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow', 'play']
    for target_idx in range(8):
        # 获取单个目标的真实标签和预测
        y_true_single = y_true[:, target_idx]
        y_pred_single = y_pred[target_idx] if isinstance(y_pred, list) else y_pred[:, target_idx]

        # 确保形状正确
        y_pred_single = y_pred_single.flatten()
        y_true_single = y_true_single.flatten()

        # 检查是否有两个类别
        unique_classes = np.unique(y_true_single)
        if len(unique_classes) > 1:
            try:
                auc = roc_auc_score(y_true_single, y_pred_single)
                auc_scores.append(auc)
                print(f"目标{target_names[target_idx]} AUC: {auc:.4f}")
            except ValueError as e:
                print(f"目标{target_names[target_idx]} AUC计算失败: {e}")
                auc_scores.append(np.nan)
        else:
            print(f"目标{target_idx + 1} 只有一个类别 ({unique_classes[0]})，无法计算AUC")
            auc_scores.append(np.nan)

    # 计算平均AUC（跳过无效值）
    valid_aucs = [auc for auc in auc_scores if not np.isnan(auc)]
    if valid_aucs:
        mean_auc = np.mean(valid_aucs)
        print(f"\n平均AUC: {mean_auc:.4f}")
        print(f"有效目标数: {len(valid_aucs)}/{len(auc_scores)}")

    return auc_scores
#================================================= Embedding Weight =================================================
dim_n = 150
emb_m = "vec"
user_feed_emb = embedding_mat("user_feed",dim_n,emb_m)
user_author_emb = embedding_mat("user_author",dim_n,emb_m)
feed_user_emb = embedding_mat("feed_user",dim_n,emb_m)
author_user_emb = embedding_mat("author_user",dim_n,emb_m)

dim_n = 128
emb_m = "vec"
user_key1_emb = embedding_mat("user_key1",dim_n,emb_m)
feed_key1_emb = embedding_mat("feed_key1",dim_n,emb_m)

user_key2_emb = embedding_mat("user_key2",dim_n,emb_m)
feed_key2_emb = embedding_mat("feed_key2",dim_n,emb_m)

user_tag_emb = embedding_mat("user_tag",32,emb_m)
feed_tag_emb = embedding_mat("feed_tag",32,emb_m)

feed_emb = embedding_mat("feed_emb",150,emb_m)

dim_n = 150
emb_m = "d2v"
user_feed_d2v = embedding_mat("user_feed",dim_n,emb_m)
user_author_d2v = embedding_mat("user_author",dim_n,emb_m)
feed_user_d2v = embedding_mat("feed_user",dim_n,emb_m)
author_user_d2v = embedding_mat("author_user",dim_n,emb_m)

data_set = user_action[["userid","feedid"]+label_cols].drop_duplicates(subset=["userid","feedid"],keep="last").reset_index(drop=True)
data_set = data_set.merge(feed_info[["feedid","authorid","videoplayseconds"]], how='left',on="feedid").fillna(0)
data_set["play"] = data_set["play"]/1000
data_set["play"] = data_set["play"]/data_set["videoplayseconds"]
data_set["play"] = data_set["play"].apply(lambda x:1 if x>0.9 else 0)
data_set = data_set.astype(int)
print(data_set)

data_userid = pad_seq(data_set, "userid", mode="data")
data_feedid = pad_seq(data_set, "feedid", mode="data")
data_authorid = pad_seq(data_set, "authorid", mode="data")
data_feats = np.concatenate([data_userid,data_feedid,data_authorid],axis=1)
data_labels = data_set[label_cols].values.reshape(-1,8)

#================================================= Train arguments =================================================
name = "m1v128k10"
phase_feat = "semi"
epochs_ = 10
kfolds_ = 10
lr_rate = 1e-3
batchsize_ = 1024 * 2

if not os.path.exists(path + "/model"):
    os.mkdir(path + "/model")

if not os.path.exists(path + "/model/" + name):
    os.mkdir(path + "/model/" + name)

the_path = path + "/model/{name}/{flag}".format(name=name, flag=phase_feat)
if not os.path.exists(the_path):
    os.mkdir(the_path)
#================================================= Dataset Preprosess =================================================

train_feats, test_feats, train_labels, test_labels = train_test_split(data_feats, data_labels, test_size=0.1,
                                                                            random_state=42)

# =========================================== Training =================================================
model = DotaModelM1V128K10(
    user_feed_emb,user_author_emb,user_tag_emb,
    user_key1_emb,user_key2_emb,feed_user_emb,feed_tag_emb,
    feed_key1_emb,feed_key2_emb,feed_emb,author_user_emb,user_feed_d2v,
    user_author_d2v,feed_user_d2v,author_user_d2v
)
csv_logger = CSVLogger(the_path + '/log_info.log')
metrics_dict={
    'output_1': tf.keras.metrics.AUC(name='auc'),
    'output_2': tf.keras.metrics.AUC(name='auc'),
    'output_3': tf.keras.metrics.AUC(name='auc'),
    'output_4': tf.keras.metrics.AUC(name='auc'),
    'output_5': tf.keras.metrics.AUC(name='auc'),
    'output_6': tf.keras.metrics.AUC(name='auc'),
    'output_7': tf.keras.metrics.AUC(name='auc'),
    'output_8': tf.keras.metrics.AUC(name='auc')
}
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
              loss=[tf.keras.losses.binary_crossentropy for i in range(8)],
              loss_weights=[1, 1, 1, 1, 1, 1, 1, 0.1],
              metrics=metrics_dict) #每一步训练时候的auc

model.fit(train_feats,
          [train_labels[:,i] for i in range(8)],
          batch_size=batchsize_,
          epochs = epochs_,
          callbacks=[csv_logger,PrintBatchInfo()],
          verbose=2)

# =========================================== Test =================================================
test_preds = model.predict(test_feats, batch_size=batchsize_, verbose=0)
testauc_scores = calculate_auc(test_labels, test_preds)
model.save(path+'/model/',save_format="tf")
    