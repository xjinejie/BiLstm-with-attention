from utils.data import Vocab, YelpDataset, load_data
from model.Bilstm_attention import BiLSTMAttnModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class Config:
    def __init__(self, overrides=None):
        cfg = {
            # 路径设置
            'train_path': 'dataset/yelp_academic_dataset_review.json',
            'test_path': 'dataset/test.json',

            # 数据处理参数
            'max_vocab_size': 30000,
            'max_len': 200,  # 句子截断/填充长度
            'min_freq': 5,
            'batch_size': 64,

            # 训练参数
            'epochs': 5,       # 演示用5轮，实际可设更多
            'learning_rate': 1e-3,
            'device': 'cuda',
            'seed': 42,

            # --- 可调整的超参数 (用于实验分析) ---
            'embed_dim': 128,
            'hidden_dim': 128,     # LSTM隐藏层维度
            'n_layers': 1,         # LSTM层数 (深度)
            'dropout': 0.3,        # Dropout概率
            'use_layer_norm': False, # 是否使用层归一化
            'use_residual': False,   # 是否使用残差连接
            'lr_decay': False,       # 是否使用学习率衰减
            'num_classes': 5,      # Yelp通常是1-5星
        }

        if overrides:
            cfg.update(overrides)

        # device_value = cfg.get('device')
        # cfg['device'] = torch.device(device_value)
        

        # 利用字典对实例属性进行初始化
        for k, v in cfg.items():
            setattr(self, k, v)
    
    # 对返回的实例属性进行字符串表示，方便打印
    def __str__(self):
        return str(self.__dict__)


    # 将配置转换为字典
    def to_dict(self):
        return dict(self.__dict__)


    # 克隆当前配置并应用更新，返回新的配置实例
    def clone_with_updates(self, updates):
        base = self.to_dict()
        base.update(updates or {})
        return Config(base)


    # 从YAML文件加载配置
    @classmethod
    def from_yaml(cls, path):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("需要先安装 PyYAML: pip install pyyaml") from exc

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        base_cfg = data.get('base', {})
        experiments = data.get('experiments', {})
        return cls(base_cfg), experiments

# 设置随机种子以保证复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_evaluate_ddp(rank, world_size, config, train_dataset, val_dataset, vocab_size, verbose=True):
    device = torch.device(f'cuda:{rank}')
    config.device = device
    
    # DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = BiLSTMAttnModel(vocab_size, config).to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 学习率衰减
    scheduler = None
    if config.lr_decay:
        # 使用 ReduceLROnPlateau: 当验证集 loss 不再下降时降低学习率
        # mode='min': 监控指标越小越好 (loss)
        # factor=0.5: 每次衰减为原来的 0.5
        # patience=1: 容忍 1 个 epoch 指标不改善
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_val_acc = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if rank == 0 and verbose:
        print(f"\n>>> Start Training: Residual={config.use_residual}, Drop={config.dropout}, Depth={config.n_layers}, LN={config.use_layer_norm}, LR_Decay={config.lr_decay}")

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        # Train
        model.train() # 训练模式
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Aggregate metrics
        train_metrics = torch.tensor([train_loss, correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        
        avg_train_loss = train_metrics[0].item() / (len(train_loader) * world_size)
        train_acc = train_metrics[1].item() / train_metrics[2].item()

        # Validation
        model.eval() # 评估模式，关闭dropout，norm使用训练数据的均值和方差
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_metrics = torch.tensor([val_loss, val_correct, val_total], device=device, dtype=torch.float64)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
        avg_val_loss = val_metrics[0].item() / (len(val_loader) * world_size)
        val_acc = val_metrics[1].item() / val_metrics[2].item()
        
        # 更新学习率 (ReduceLROnPlateau 需要传入验证集 loss)
        if config.lr_decay and scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if rank == 0:
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.module.state_dict()
    
    return best_model_state, best_val_acc, history

def plot_results(all_histories, save_dir="saved_models/figures"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 为每个实验画 Loss 和 Acc
    for exp_name, history in all_histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        plt.figure()
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.title(f'{exp_name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{exp_name}_loss.png'))
        plt.close()
        
        # Acc
        plt.figure()
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.title(f'{exp_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{exp_name}_acc.png'))
        plt.close()

    # 2. 画所有实验的 Val Acc 对比
    plt.figure()
    for exp_name, history in all_histories.items():
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=exp_name)
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'comparison_val_acc.png'))
    plt.close()

def main_worker(rank, world_size, base_config, experiments, train_dataset, val_dataset, test_dataset, vocab_size):
    setup(rank, world_size)
    
    results = {}
    all_histories = {}
    best_model_state = None
    best_val_acc = 0
    
    if rank == 0:
        print("\n================ 实验开始 ================")
    
    for exp_name, changes in experiments.items():
        current_config = base_config.clone_with_updates(changes)
        
        if rank == 0:
            print(f"\n运行实验: [{exp_name}]")
        
        model_state, acc, history = train_evaluate_ddp(rank, world_size, current_config, train_dataset, val_dataset, vocab_size)
        
        if rank == 0:
            results[exp_name] = acc
            all_histories[exp_name] = history
            print("\n-----------------------------------------------")
            print(f"Result [{exp_name}]: best val Top-1 Acc = {acc:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = model_state
    

    if rank == 0:
        # 绘图
        plot_results(all_histories)

        # 3. 最终总结
        print("\n================ 最终总结 ================")
        print(f"{'实验名称':<20} | {'最佳验证准确率'}")
        print("-" * 40)
        best_exp = ""
        best_val_acc = 0
        for name, val_acc in results.items():
            print(f"{name:<20} | {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_exp = name
                
        print("-" * 40)
        print(f"Best Configuration: {best_exp} with val_Acc: {best_val_acc:.4f}")

        if best_model_state:
            torch.save(best_model_state,f"saved_models/models/{best_exp}_best_val_acc.pth")

            # Evaluate on Test Set
            print(f"\nEvaluating Best Model ({best_exp}) on Test Set...")
            best_config = base_config.clone_with_updates(experiments[best_exp])
            best_config.device = torch.device(f'cuda:{rank}')
            
            model = BiLSTMAttnModel(vocab_size, best_config).to(best_config.device)
            model.load_state_dict(best_model_state)
            model.eval()
            
            test_loader = DataLoader(test_dataset, batch_size=best_config.batch_size, shuffle=False, num_workers=4)
            
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(best_config.device), labels.to(best_config.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = test_correct / test_total
            print(f"Test Accuracy: {test_acc:.4f}")
            
    cleanup()
            

if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    try:
        base_config, experiments = Config.from_yaml(CONFIG_PATH)
        print(f"从 {CONFIG_PATH} 读取配置。")
    except FileNotFoundError:
        print(f"未找到 {CONFIG_PATH}，使用代码内默认配置。")
        base_config = Config()
        experiments = {
            "Baseline": {},
            "High_Dropout": {"dropout": 0.8},
            "Deep_Network": {"n_layers": 3},
            "Deep_ResNet": {"n_layers": 3, "use_residual": True},
            "With_LayerNorm": {"use_layer_norm": True},
            "With_LR_Decay": {"lr_decay": True, "epochs": 8}
        }
    except ImportError as exc:
        print(exc)
        exit(1)

    # 1. 准备数据 (仅做一次)
    set_seed(base_config.seed)
    
    # 限制数据量以快速演示，实际运行时请设为 None 或更大值
    DEBUG_LIMIT = None 
    
    raw_train_data = load_data(base_config.train_path, limit=DEBUG_LIMIT)
    raw_test_data = load_data(base_config.test_path, limit=None) # Test集一般较小，全部加载
    
    if not raw_train_data:
        print("请检查 dataset 目录下是否存在数据文件。")
        exit()

    # 构建词表
    all_text = [d[0] for d in raw_train_data]
    vocab = Vocab(all_text, base_config.max_vocab_size, base_config.min_freq)
    print(f"Vocab Size: {len(vocab)}")
    
    # 划分训练/验证 (9:1)
    dataset_full = YelpDataset(raw_train_data, vocab, base_config.max_len)
    train_size = int(0.9 * len(dataset_full))
    val_size = len(dataset_full) - train_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])
    
    test_dataset = YelpDataset(raw_test_data, vocab, base_config.max_len)

    # 2. 定义实验配置列表 (对比实验)
    if not experiments:
        experiments = {"Baseline": {}}
    
    world_size = 8
    mp.spawn(main_worker, args=(world_size, base_config, experiments, train_dataset, val_dataset, test_dataset, len(vocab)), nprocs=world_size, join=True)