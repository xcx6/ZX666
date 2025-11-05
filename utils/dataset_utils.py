import os
import types
from collections import defaultdict

import ujson
import numpy as np
import json
import torch
import random


def check(config_path, train_path, test_path, num_clients, num_labels, niid=False,
        real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_labels'] == num_labels and \
            config['non_iid'] == niid and \
            config['real_world'] == real and \
            config['partition'] == partition:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def read_record(file):
    with open(file,"r") as f:
        dataJson = json.load(f)
        users_train = dataJson["train_data"]
        #users_test = dataJson["test_data"]
    dict_users_train = {}
    #dict_users_test = {}
    for key,value in users_train.items():
        newKey = int(key)
        dict_users_train[newKey] = value
    '''
    for key,value in users_test.items():
        newKey = int(key)
        dict_users_test[newKey] = value
    '''
    return dict_users_train #, dict_users_test

def separate_data(train_data, num_clients, num_classes, beta=0.4):


    y_train = np.array([i[1] for i in train_data])

    min_size_train = 0
    min_require_size = 10
    K = num_classes

    N_train = len(y_train)
    dict_users_train = {}

    while min_size_train < min_require_size:
        idx_batch_train = [[] for _ in range(num_clients)]
        idx_batch_test = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k_train = np.where(y_train == k)[0]
            np.random.shuffle(idx_k_train)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions_train = np.array([p * (len(idx_j) < N_train / num_clients) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break

    for j in range(num_clients):
        np.random.shuffle(idx_batch_train[j])
        dict_users_train[j] = idx_batch_train[j]

    train_cls_counts = record_net_data_stats(y_train,dict_users_train)

    return dict_users_train

def separate_data_clustered(train_data, num_clients, num_classes, beta=0.4, num_clusters=None, cluster_data_ratios=None):
    """
    生成簇内同分布的Non-IID数据
    每个簇内的客户端拥有相同的主要数据类型，但数据分布相同
    支持不同簇分配不同数量的数据
    
    Args:
        train_data: 训练数据
        num_clients: 客户端数量
        num_classes: 类别数量
        beta: Dirichlet分布参数，控制异构程度
        num_clusters: 簇数量，默认为类别数量
        cluster_data_ratios: 每个簇的数据量比例列表，如[1.0, 0.5, 0.3, 0.2]表示簇0分配100%数据，簇1分配50%，等等
    
    Returns:
        dict_users_train: 客户端数据分配字典
        client_cluster_map: 客户端到簇的映射字典 {client_id: cluster_id}
    """
    y_train = np.array([i[1] for i in train_data])
    
    if num_clusters is None:
        num_clusters = num_classes
    
    # 确保簇数量不超过类别数量
    num_clusters = min(num_clusters, num_classes)
    
    # 设置默认的数据量比例（如果未提供）
    if cluster_data_ratios is None:
        cluster_data_ratios = [1.0] * num_clusters  # 默认每个簇分配相同比例
    else:
        # 确保比例列表长度与簇数量匹配
        if len(cluster_data_ratios) != num_clusters:
            print(f"警告: cluster_data_ratios长度({len(cluster_data_ratios)})与簇数量({num_clusters})不匹配，使用默认比例")
            cluster_data_ratios = [1.0] * num_clusters
    
    # 计算每个簇的客户端数量
    clients_per_cluster = num_clients // num_clusters
    remaining_clients = num_clients % num_clusters
    
    print(f"生成簇内同分布数据:")
    print(f"  总客户端数: {num_clients}")
    print(f"  簇数量: {num_clusters}")
    print(f"  每簇客户端数: {clients_per_cluster}")
    print(f"  剩余客户端数: {remaining_clients}")
    print(f"  簇数据量比例: {cluster_data_ratios}")
    
    # 为每个簇分配主要类别
    cluster_main_classes = list(range(num_clusters))
    
    # 为每个类别收集数据索引
    class_indices = {}
    for k in range(num_classes):
        class_indices[k] = np.where(y_train == k)[0]
        np.random.shuffle(class_indices[k])
    
    dict_users_train = {}
    client_cluster_map = {}  # 客户端到簇的映射
    client_idx = 0
    
    # 为每个簇分配数据
    for cluster_id in range(num_clusters):
        main_class = cluster_main_classes[cluster_id]
        cluster_size = clients_per_cluster + (1 if cluster_id < remaining_clients else 0)
        
        # 计算该簇的数据量比例
        cluster_ratio = cluster_data_ratios[cluster_id]
        
        print(f"  簇 {cluster_id}: 主要类别 {main_class}, 客户端数 {cluster_size}, 数据量比例 {cluster_ratio}")
        
        # 为该簇的客户端分配数据
        for client_in_cluster in range(cluster_size):
            client_data_indices = []
            
            # 为主要类别分配大部分数据
            main_class_data = class_indices[main_class]
            main_class_ratio = 0.8  # 80%的数据来自主要类别
            
            # 计算该簇每个客户端的数据量（基于比例）
            base_samples_per_client = len(y_train) // num_clients
            total_samples_per_client = int(base_samples_per_client * cluster_ratio)
            main_class_samples = int(total_samples_per_client * main_class_ratio)
            
            # 分配主要类别数据
            if len(main_class_data) >= main_class_samples:
                selected_main = np.random.choice(main_class_data, main_class_samples, replace=False)
                client_data_indices.extend(selected_main.tolist())
                # 从可用数据中移除已选择的
                class_indices[main_class] = np.setdiff1d(main_class_data, selected_main)
            else:
                # 如果主要类别数据不够，分配所有可用的主要类别数据
                if len(main_class_data) > 0:
                    client_data_indices.extend(main_class_data.tolist())
                    class_indices[main_class] = np.array([])
            
            # 为其他类别分配剩余数据
            remaining_samples = total_samples_per_client - len(client_data_indices)
            if remaining_samples > 0:
                other_classes = [k for k in range(num_classes) if k != main_class]
                other_class_samples = remaining_samples // len(other_classes)
                
                for other_class in other_classes:
                    if len(class_indices[other_class]) >= other_class_samples:
                        selected_other = np.random.choice(class_indices[other_class], other_class_samples, replace=False)
                        client_data_indices.extend(selected_other.tolist())
                        class_indices[other_class] = np.setdiff1d(class_indices[other_class], selected_other)
                    else:
                        # 如果某个类别的数据不够，分配所有可用的数据
                        if len(class_indices[other_class]) > 0:
                            client_data_indices.extend(class_indices[other_class].tolist())
                            class_indices[other_class] = np.array([])
                
                # 如果还有剩余样本需要分配，从所有可用类别中随机选择
                still_remaining = total_samples_per_client - len(client_data_indices)
                if still_remaining > 0:
                    available_classes = [k for k in range(num_classes) if len(class_indices[k]) > 0]
                    if available_classes:
                        for _ in range(still_remaining):
                            if not available_classes:
                                break
                            # 随机选择一个有数据的类别
                            chosen_class = np.random.choice(available_classes)
                            if len(class_indices[chosen_class]) > 0:
                                selected_sample = np.random.choice(class_indices[chosen_class], 1, replace=False)
                                client_data_indices.extend(selected_sample.tolist())
                                class_indices[chosen_class] = np.setdiff1d(class_indices[chosen_class], selected_sample)
                                # 如果这个类别没有数据了，从可用类别列表中移除
                                if len(class_indices[chosen_class]) == 0:
                                    available_classes.remove(chosen_class)
            
            # 随机打乱数据
            np.random.shuffle(client_data_indices)
            dict_users_train[client_idx] = client_data_indices
            client_cluster_map[client_idx] = cluster_id  # 记录客户端所属簇
            client_idx += 1
    
    # 记录数据统计信息
    train_cls_counts = record_net_data_stats(y_train, dict_users_train)
    
    # 打印客户端簇归属信息
    print(f"\n客户端簇归属映射:")
    for client_id, cluster_id in client_cluster_map.items():
        print(f"  客户端 {client_id} -> 簇 {cluster_id} (主要类别: {cluster_main_classes[cluster_id]})")
    
    # 打印每个簇的详细数据分布信息
    print_cluster_distribution_info(train_cls_counts, client_cluster_map, cluster_main_classes, num_classes)
    
    return dict_users_train, client_cluster_map

def print_cluster_distribution_info(train_cls_counts, client_cluster_map, cluster_main_classes, num_classes):
    """
    打印每个簇的详细数据分布信息
    
    Args:
        train_cls_counts: 客户端数据统计信息 {client_id: {class_id: count}}
        client_cluster_map: 客户端到簇的映射 {client_id: cluster_id}
        cluster_main_classes: 每个簇的主要类别 [cluster_id: main_class]
        num_classes: 总类别数
    """
    print(f"\n=== 簇数据分布详细信息 ===")
    
    # 按簇分组客户端
    cluster_clients = {}
    for client_id, cluster_id in client_cluster_map.items():
        if cluster_id not in cluster_clients:
            cluster_clients[cluster_id] = []
        cluster_clients[cluster_id].append(client_id)
    
    # 为每个簇打印详细信息
    for cluster_id in sorted(cluster_clients.keys()):
        main_class = cluster_main_classes[cluster_id]
        clients_in_cluster = cluster_clients[cluster_id]
        
        print(f"\n簇 {cluster_id} (主要类别: {main_class}):")
        print(f"  客户端: {clients_in_cluster}")
        
        # 计算簇内每个类别的总数据量
        cluster_class_counts = {}
        cluster_total_samples = 0
        
        for client_id in clients_in_cluster:
            client_data = train_cls_counts[client_id]
            for class_id, count in client_data.items():
                if class_id not in cluster_class_counts:
                    cluster_class_counts[class_id] = 0
                cluster_class_counts[class_id] += count
                cluster_total_samples += count
        
        # 打印簇内类别分布
        print(f"  簇内数据分布:")
        for class_id in range(num_classes):
            count = cluster_class_counts.get(class_id, 0)
            percentage = (count / cluster_total_samples * 100) if cluster_total_samples > 0 else 0
            marker = " ★" if class_id == main_class else ""
            print(f"    类别 {class_id}: {count:4d} 样本 ({percentage:5.1f}%){marker}")
        
        # 打印每个客户端的详细分布
        print(f"  各客户端详细分布:")
        for client_id in clients_in_cluster:
            client_data = train_cls_counts[client_id]
            client_total = sum(client_data.values())
            print(f"    客户端 {client_id:2d}: ", end="")
            for class_id in range(num_classes):
                count = client_data.get(class_id, 0)
                percentage = (count / client_total * 100) if client_total > 0 else 0
                marker = "★" if class_id == main_class else " "
                print(f"[{class_id}:{count:3d}({percentage:4.1f}%){marker}] ", end="")
            print(f"总计:{client_total}")
    
    print(f"\n=== 簇分布统计摘要 ===")
    for cluster_id in sorted(cluster_clients.keys()):
        main_class = cluster_main_classes[cluster_id]
        clients_in_cluster = cluster_clients[cluster_id]
        
        # 计算主要类别的平均占比
        main_class_ratios = []
        for client_id in clients_in_cluster:
            client_data = train_cls_counts[client_id]
            client_total = sum(client_data.values())
            main_class_count = client_data.get(main_class, 0)
            ratio = (main_class_count / client_total * 100) if client_total > 0 else 0
            main_class_ratios.append(ratio)
        
        avg_main_ratio = np.mean(main_class_ratios)
        std_main_ratio = np.std(main_class_ratios)
        
        print(f"簇 {cluster_id}: {len(clients_in_cluster):2d}个客户端, "
              f"主要类别{main_class}平均占比: {avg_main_ratio:5.1f}% ± {std_main_ratio:4.1f}%")

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():

        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))

    return net_cls_counts

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
                num_labels, statistic, niid=False, real=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_labels': num_labels,
        'non_iid': niid,
        'real_world': real,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
    }

    # gc.collect()

    for idx, train_dict in enumerate(train_data):
        with open(train_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(train_dict, f)
    for idx, test_dict in enumerate(test_data):
        with open(test_path[:-5] + str(idx)  + '_' + '.json', 'w') as f:
            ujson.dump(test_dict, f)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list

def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = {i: [] for i in range(num_users)}
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

def gen_random_loaders(dataset, num_users, rand_set_all = None, classes_per_user=2):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    if rand_set_all is None:
        rand_set_all = gen_classes_per_node(dataset, num_users, classes_per_user)

    usr_subset_idx = gen_data_split(dataset, num_users, rand_set_all)

    #cls_counts = record_net_data_stats(dataset.targets, usr_subset_idx)

    return usr_subset_idx,rand_set_all