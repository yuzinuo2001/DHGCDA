import torch
import numpy as np
from model_GPU import *
from sklearn import metrics
import pickle
import random
import csv
                                
import matplotlib
matplotlib.use("Agg")                
import matplotlib.pyplot as plt
import time
import copy
import os
import json
import platform
import sys
import hashlib
from datetime import datetime
import datetime

print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda_available:", torch.cuda.is_available())
print("[CHECK] cuda_device_count:", torch.cuda.device_count())
print("[CHECK] current_device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("[CHECK] device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
                    
                                     
                
now = datetime.datetime.now()

                                 
                                 
                                 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPRO_DIR = os.path.join(BASE_DIR, "repro_runs")
RUN_CONFIG_PATH = os.path.join(REPRO_DIR, "last_run_config.json")

                                                            
USE_SAVED_RUN_CONFIG = False


def _sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int, deterministic: bool = True) -> None:




                            
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

                                  
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
                                        
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_run_id(seed_base: int) -> str:
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{t}_seed{seed_base}"


def build_run_config(local_vars: dict, seed_base: int, deterministic: bool, run_id: str) -> dict:



    cfg = {
        "run_id": run_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed_base": int(seed_base),
        "deterministic": bool(deterministic),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

            
    try:
        import torch
        cfg["torch_version"] = torch.__version__
        cfg["cuda_version"] = getattr(torch.version, "cuda", None)
        cfg["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass

    try:
        import numpy as np
        cfg["numpy_version"] = np.__version__
    except Exception:
        pass

                            
    keep_keys = [
        "device", "epochs", "lr", "bio_out_dim", "hgnn_dim",
        "weight_decay", "scheduler_step_size", "scheduler_gamma",
        "CMI_data", "USE_TEMPERATURE_SCALING",
    ]
    for k in keep_keys:
        if k in local_vars:
            v = local_vars[k]
            if isinstance(v, (int, float, str, bool, type(None))):
                cfg[k] = v

                     
    try:
        cfg["code_sha1"] = {
            "main_GPU.py": _sha1_of_file(os.path.join(BASE_DIR, "main_GPU.py")),
            "model_GPU.py": _sha1_of_file(os.path.join(BASE_DIR, "model_GPU.py")),
        }
    except Exception:
        pass

    return cfg




csv.field_size_limit(500 * 1024 * 1024)

USE_TEMPERATURE_SCALING = False

def prepare_hyg_edges(G_d, G_c, G_cc, G_dd, G_mc, G_md, n_circ, m_dis, device):











    def _fix_edge_index(edge_idx, num_nodes, name):



        edge_idx = torch.as_tensor(edge_idx, dtype=torch.long, device=device)

        node_idx = edge_idx[0]
        hyper_idx = edge_idx[1]

        print(
            f"[DEBUG] {name} node_idx min={node_idx.min().item()}, "
            f"max={node_idx.max().item()}, num_nodes={num_nodes}"
        )
        print(
            f"[DEBUG] {name} hyper_idx min={hyper_idx.min().item()}, "
            f"max={hyper_idx.max().item()}"
        )

                                                   
        if node_idx.min().item() == 1 and node_idx.max().item() == num_nodes:
            print(f"[INFO] {name}: 检测到 1-based 节点索引，自动转为 0-based")
            node_idx = node_idx - 1
            edge_idx[0] = node_idx

                            
        if node_idx.min().item() < 0 or node_idx.max().item() >= num_nodes:
            raise ValueError(
                f"{name}: 节点索引越界! "
                f"min={node_idx.min().item()}, max={node_idx.max().item()}, "
                f"num_nodes={num_nodes}"
            )

        return edge_idx

    def _fix_hye_node(hye_node, num_nodes, name):









        hye_node = torch.as_tensor(hye_node, dtype=torch.long, device=device)

        mn = hye_node.min().item()
        mx = hye_node.max().item()
        print(f"[DEBUG] {name} hye_node min={mn}, max={mx}, num_nodes={num_nodes}")

                                                             
        if mn < 0 or mx > num_nodes:








            hye_node = hye_node.clone()
            hye_node[hye_node < 0] = 0
            hye_node[hye_node > num_nodes] = 0

            mn = hye_node.min().item()
            mx = hye_node.max().item()
            print(f"[DEBUG] {name} 修正后 hye_node min={mn}, max={mx}")

        return hye_node

                           
    cm_edge, cm_len, cm_node = G_d                     
    mc_edge, mc_len, mc_node = G_c                     
    cc_edge, cc_len, cc_node = G_cc                     
    mm_edge, mm_len, mm_node = G_dd                     

                          
    cm_edge_t = _fix_edge_index(cm_edge, m_dis, "hy_cm_edge")
    cm_node_t = _fix_hye_node(cm_node, m_dis, "hy_cm_node")

                       
    mc_edge_t = _fix_edge_index(mc_edge, n_circ, "hy_mc_edge")
    mc_node_t = _fix_hye_node(mc_node, n_circ, "hy_mc_node")

    cc_edge_t = _fix_edge_index(cc_edge, n_circ, "hy_cc_edge")
    cc_node_t = _fix_hye_node(cc_node, n_circ, "hy_cc_node")

                           
    mm_edge_t = _fix_edge_index(mm_edge, m_dis, "hy_mm_edge")
    mm_node_t = _fix_hye_node(mm_node, m_dis, "hy_mm_node")

                                                          
    hy_cm_t = [cm_edge_t, cm_len, cm_node_t]
    hy_mc_t = [mc_edge_t, mc_len, mc_node_t]
    hy_cc_t = [cc_edge_t, cc_len, cc_node_t]
    hy_mm_t = [mm_edge_t, mm_len, mm_node_t]
    if G_mc is not None:
        mi_edge, mi_len, mi_node = G_mc
        mi_edge_t = _fix_edge_index(mi_edge, n_circ, "hy_mi_edge")
        mi_node_t = _fix_hye_node(mi_node, n_circ, "hy_mi_node")
        hy_mi_t = [mi_edge_t, mi_len, mi_node_t]
    else:
        hy_mi_t = None
    if G_md is not None:
        do_edge, do_len, do_node = G_md
        do_edge_t = _fix_edge_index(do_edge, m_dis, "hy_do_edge")
        do_node_t = _fix_hye_node(do_node, m_dis, "hy_do_node")
        hy_do_t = [do_edge_t, do_len, do_node_t]
    else:
        hy_do_t = None


    return [hy_cm_t, hy_mc_t, hy_cc_t, hy_mm_t, hy_mi_t, hy_do_t]


      
def plot_training_progress(fold_auc_list):
                  
    mean_auc = np.mean(np.array(fold_auc_list), axis=0)
    plt.figure(figsize=(6, 4))
    plt.plot(mean_auc, marker='o')
    plt.title("5-Fold Mean AUC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_curve(values, title, ylabel, metric_name):




    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(6, 4))

    final_val = values[-1] if len(values) > 0 else 0.0
    label = f"{metric_name} (final={final_val:.4f})"

    plt.plot(epochs, values, marker='o', label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()                  
    plt.tight_layout()
    plt.show()

def split_by_fold(values, num_folds, epochs):





    import numpy as np
    import torch

    clean = []
    for v in values:
        if isinstance(v, torch.Tensor):
            clean.append(float(v.detach().cpu().item()))
        else:
            clean.append(float(v))

    values = np.array(clean, dtype=float)
    total = num_folds * epochs

    if values.size == 0:
        raise ValueError(f"[split_by_fold] 列表为空，无法按折拆分。")

    if values.size < total:
        pad_len = total - values.size
        pad = np.full(pad_len, values[-1], dtype=float)
        values = np.concatenate([values, pad])
    elif values.size > total:
        values = values[:total]

    return values.reshape(num_folds, epochs)



def plot_metric_mean_std(*args, **kwargs):













    import numpy as np
    import matplotlib.pyplot as plt

                                          
    ax = kwargs.pop("ax", None)
    title = kwargs.pop("title", "")
    ylabel = kwargs.pop("ylabel", "")
    metric_name = kwargs.pop("metric_name", None)

                 
    smooth_alpha = kwargs.pop("smooth_alpha", 0.20)                                  
    smooth_window = kwargs.pop("smooth_window", 5)                            

    epochs = kwargs.pop("epochs", None)                  

                                                  
    values_folds = None

    if len(args) >= 1:
        first = args[0]
        is_axes = hasattr(first, "plot") and hasattr(first, "set_title")                        
        if is_axes:
                                                                 
            ax = first if ax is None else ax
            if len(args) >= 2:
                epochs = args[1]
            if len(args) >= 3:
                values_folds = args[2]
            if len(args) >= 4 and not title:
                title = args[3]
            if len(args) >= 5 and not ylabel:
                ylabel = args[4]
        else:
                                                   
            values_folds = first

    if values_folds is None:
        raise TypeError("plot_metric_mean_std: 缺少 values_folds（第一个位置参数）。")

    if ax is None:
        ax = plt.gca()

                                     
    def _ema(x, alpha):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
        return y

    def _roll_mean(x, w):
        x = np.asarray(x, dtype=float)
        if w is None or w <= 1 or x.size < 2:
            return x
        w = int(w)
        w = max(2, min(w, x.size))        
        kernel = np.ones(w, dtype=float) / float(w)
                                       
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        xp = np.pad(x, (pad_left, pad_right), mode="edge")
        return np.convolve(xp, kernel, mode="valid")

                                                             
                       
                                    
                         
    mat = None

    if isinstance(values_folds, np.ndarray):
        if values_folds.ndim == 1:
            mat = values_folds[None, :]
        else:
            mat = values_folds
    else:
                                 
        vals = []
        for v in values_folds:
            if v is None:
                continue
            vals.append(np.asarray(v, dtype=float))
        if len(vals) == 0:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            return
        min_len = min(len(v) for v in vals)
        vals = [v[:min_len] for v in vals]
        mat = np.stack(vals, axis=0)

                   
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)

    n_folds, T = mat.shape

                               
    if epochs is None:
        x = np.arange(1, T + 1)
    else:
        x = np.asarray(list(epochs), dtype=float)[:T]
        if x.size < T:
                   
            x = np.arange(1, T + 1)

                                                         
    mat_s = np.empty_like(mat)
    for i in range(n_folds):
        y = mat[i]
        if smooth_alpha is not None and smooth_alpha > 0:
            y = _ema(y, float(smooth_alpha))
        y = _roll_mean(y, smooth_window)
        mat_s[i] = y

    mean = mat_s.mean(axis=0)
    std = mat_s.std(axis=0)

                                      
          
    for i in range(n_folds):
        ax.plot(x, mat_s[i], linewidth=1, alpha=0.20)

                   
    name = metric_name if metric_name is not None else (ylabel if ylabel else "Metric")
    ax.plot(x, mean, linewidth=2.5, label=f"{name} (final mean={mean[-1]:.4f})")
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, label="±1 std")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()



def ReadMyCsv4(SavaDict, fileName):
    csv_reader = csv.reader(open(fileName))
    count = 0
    for row in csv_reader:
        SavaDict[row[0]] = count
        count = count + 1
    return


def ReadMyCsv5(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        c = int(row[0]) - 1
        d = disease_to_idx[row[1]]                                             
        cd = [c, d]
        SaveList.append(cd)
    return


def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


def MyLabel_val_test(valSample, testSample):
    label = []
    for i in range(int(len(valSample) / 2)):
        label.append(1)
    for i in range(int(len(testSample) / 2)):
        label.append(1)
    for i in range(int(len(valSample) / 2)):
        label.append(0)
    for i in range(int(len(testSample) / 2)):
        label.append(0)
    return label


def load_hyg(filename):
    with open(filename, 'rb') as f:
        hyg_dict = pickle.load(f)

    G_d = hyg_dict['hy_cm']
    G_c = hyg_dict['hy_mc']
    G_cc = hyg_dict['hy_cc_top5']
    G_dd = hyg_dict['hy_mm_top5']
    G_mc = hyg_dict.get('hy_mi', None)
    G_md = hyg_dict.get('hy_do', None)


    S_cc = hyg_dict['lg_c2']                 
    S_dd = hyg_dict['lg_m2']               

    return (
        G_d,
        G_c,
        G_cc,
        G_dd,
        G_mc,
        G_md,
        torch.from_numpy(S_cc).type(torch.FloatTensor),
        torch.from_numpy(S_dd).type(torch.FloatTensor),
    )
def get_rec_weights(epoch, warm_epochs=20, ramp_epochs=10, w_circ_final=0.01, w_dis_final=0.02):




    if epoch < warm_epochs:
        return 0.0, 0.0

    t = min(1.0, (epoch - warm_epochs) / max(1, ramp_epochs))
    return w_circ_final * t, w_dis_final * t

                                                               
                                                                          
                                 
REC_W_CIRC_FINAL = 0.02
REC_W_DIS_FINAL  = 0.05

                
                
     
                                            
                                            
                               
                              
                                                                
ABLATION_PRESETS = {
    "full": dict(drop_cc=False, drop_dd=False, drop_mi=False, drop_do=False),

            
    "wocc": dict(drop_cc=True,  drop_dd=False, drop_mi=False, drop_do=False),
    "wodd": dict(drop_cc=False, drop_dd=True,  drop_mi=False, drop_do=False),
    "womi": dict(drop_cc=False, drop_dd=False, drop_mi=True,  drop_do=False),
    "wocc_wodd": dict(drop_cc=True, drop_dd=True, drop_mi=False, drop_do=False),
    "wocc_womi": dict(drop_cc=True, drop_dd=False, drop_mi=True, drop_do=False),
    "wodd_womi": dict(drop_cc=False, drop_dd=True, drop_mi=True, drop_do=False),
    "wocc_wodd_womi": dict(drop_cc=True, drop_dd=True, drop_mi=True, drop_do=False),

                                   

                                                      
                                                      
    "womv": dict(drop_cc=True, drop_dd=True, drop_mi=True, drop_do=False, womv_keep_do_as_gcd=False),

                                                       
    "wohe": dict(drop_cc=False, drop_dd=False, drop_mi=False, drop_do=False),

                                                    
    "wovw": dict(drop_cc=False, drop_dd=False, drop_mi=False, drop_do=False),
}

def _make_dummy_hy():






    edge_index = np.array([[0], [0]], dtype=np.int64)           
    hye_len = np.array([1], dtype=np.int64)                    
    hye_node = np.array([[0]], dtype=np.int64)                  
    return [edge_index, hye_len, hye_node]


def apply_ablation_to_hyg(G_d, G_c, G_cc, G_dd, G_mc, G_md, preset: dict):








    if preset.get("drop_cc", False):
        G_cc = _make_dummy_hy()
    if preset.get("drop_dd", False):
        G_dd = _make_dummy_hy()
    if preset.get("drop_mi", False):
        G_mc = None
    if preset.get("drop_do", False):
        G_md = None

                          
    if "womv_keep_do_as_gcd" in preset:
        keep_do = bool(preset.get("womv_keep_do_as_gcd", False))
        G_cc = _make_dummy_hy()
        G_dd = _make_dummy_hy()
        G_mc = None
        if not keep_do:
            G_md = None

    return G_d, G_c, G_cc, G_dd, G_mc, G_md

def apply_ablation_to_rec_weights(ablation_mode: str, preset: dict):




    global REC_W_CIRC_FINAL, REC_W_DIS_FINAL

                            
    REC_W_CIRC_FINAL = 0.02
    REC_W_DIS_FINAL  = 0.05

    if str(ablation_mode).upper() == "B":
        if preset.get("drop_cc", False):
            REC_W_CIRC_FINAL = 0.0
        if preset.get("drop_dd", False):
            REC_W_DIS_FINAL = 0.0
                                                                   


def train(circ_feat, disease_feat, pair_idx_train, y_true_train, hy_ls, S_cc, S_dd, epoch):
    model.train()
                                    
    optimizer.zero_grad()

    c_idx_train = torch.from_numpy(pair_idx_train[:, 0]).long().to(device)
    d_idx_train = torch.from_numpy(pair_idx_train[:, 1]).long().to(device)
    y_true_train_tensor = torch.from_numpy(y_true_train).float().to(device)

    _, logits_train, Z_c_train, Z_d_train = model(
        circ_feat,
        disease_feat,
        c_idx_train,
        d_idx_train,
        hy_ls
    )
    y_hat_train = torch.sigmoid(logits_train)                 
                                                                
                                                                
                                                                             
                                                                
                                                                             
    Z_c_norm = torch.nn.functional.normalize(Z_c_train, dim=1)
    Z_d_norm = torch.nn.functional.normalize(Z_d_train, dim=1)

    S_hat_cc_train = (Z_c_norm @ Z_c_norm.T + 1.0) * 0.5                     
    S_hat_dd_train = (Z_d_norm @ Z_d_norm.T + 1.0) * 0.5                     

    L_cc = loss_function2(S_hat_cc_train, S_cc)
    L_dd = loss_function2(S_hat_dd_train, S_dd)

                                                                       
    lambda_c, lambda_d = get_rec_weights(
        epoch,
        warm_epochs=35,
        ramp_epochs=15,
        w_circ_final=REC_W_CIRC_FINAL,
        w_dis_final=REC_W_DIS_FINAL
    )

    L_pred = loss_function1(
        logits_train.view(-1, 1),
        y_true_train_tensor.view(-1, 1)
    )
                                               
    loss_rec_raw = lambda_c * L_cc + lambda_d * L_dd
    if (lambda_c + lambda_d) > 0:
        bal = (L_pred.detach() / (loss_rec_raw.detach() + 1e-8)).clamp(max=1.0)
    else:
        bal = torch.tensor(0.0, device=device)



    L_rec = bal * loss_rec_raw
    L_total = L_pred + L_rec
                                         
    L_total.backward()
                                  
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

                                                                
    with torch.no_grad():
        model.eval()
        _, logits_train_eval, Z_c_eval, Z_d_eval = model(
            circ_feat, disease_feat, c_idx_train, d_idx_train, hy_ls
        )
        y_hat_train_eval = torch.sigmoid(logits_train_eval)

                                                          
        S_hat_cc_eval = Z_c_eval @ Z_c_eval.transpose(0, 1)
        S_hat_dd_eval = Z_d_eval @ Z_d_eval.transpose(0, 1)

                                     
                                           
                                           
          
                                                                                                            
                                                          
                                                          
        loss_1_log = L_pred.detach().item()
        loss_2_log = L_rec.detach().item()
        loss_train_log = float(loss_1_log + loss_2_log)
    model.train()

    score_train_cpu = y_hat_train_eval.detach().cpu().numpy().reshape(-1)
    label_train_cpu = y_true_train_tensor.detach().cpu().numpy().reshape(-1)

    auc_train = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
    acc_train = metrics.accuracy_score(label_train_cpu, score_train_cpu > 0.5)
    recall_train = metrics.recall_score(label_train_cpu, score_train_cpu > 0.5)

    train_auc_list.append(auc_train)
    train_acc_list.append(acc_train)
    train_recall_list.append(recall_train)
    train_losses.append(loss_train_log)

           
                                                                                                                                  
    print(
        f'Epoch {e + 1} - AUC: {auc_train:.4f} - ACC: {acc_train:.4f} - Recall: {recall_train:.4f} '
        f'- Loss: {loss_train_log:.4f} '
        f'- L1(BCE): {loss_1_log:.4f} - L2(rec): {loss_2_log:.4f} '
        f'- w_c: {lambda_c:.4g} - w_d: {lambda_d:.4g} - bal: {bal.item():.4g}'
    )


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-6)

def test_logits(circ_feat, disease_feat, data, hy_ls):
    with torch.no_grad():
        model.eval()
        c_idx = torch.from_numpy(data[:, 0]).long().to(device)
        d_idx = torch.from_numpy(data[:, 1]).long().to(device)
        _, logits, _, _ = model(circ_feat, disease_feat, c_idx, d_idx, hy_ls)
        return logits.detach()


def test(circ_feat, disease_feat, pair_idx_test, y_true_test, hy_ls, verbose=True):
    with torch.no_grad():
        model.eval()
                                         

        c_idx_test = torch.from_numpy(pair_idx_test[:, 0]).long().to(device)
        d_idx_test = torch.from_numpy(pair_idx_test[:, 1]).long().to(device)

        _, logits_test, _, _ = model(
            circ_feat,
            disease_feat,
            c_idx_test,
            d_idx_test,
            hy_ls
        )

        test_score_array = logits_test.detach().cpu().numpy()
        auc_test = metrics.roc_auc_score(y_true_test, test_score_array)

        pred_label_test = [1 if j > 0.5 else 0 for j in test_score_array]
        acc_test = metrics.accuracy_score(y_true_test, pred_label_test)
        if verbose:
            print(
                'AUC_test:{:.6f},'.format(auc_test),
                'ACC_test:{:.6f},'.format(acc_test)
            )

        return test_score_array


if __name__ == '__main__':

    start_time = time.time()
                                                                
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full",
                        help=f"preset in {list(ABLATION_PRESETS.keys())}")
    parser.add_argument("--ablation_mode", type=str, default="A",
                        help="A: drop channel only; B: drop channel + disable corresponding rec loss")

    parser.add_argument("--dual_ablation", type=str, default="full",
                        choices=["full", "node_only", "hyper_only", "bipartite_x2_only", "similarity_x2_only"],
                        help="Dual-channel ablation inside HgnnEncoder (mc/cm/cc/mm only; keep miRNA/DOID unchanged).")
                                
    parser.add_argument("--view_fusion", type=str, default="attn",
                        choices=["attn", "mean"],
                        help="model-side view fusion: attn=FusionAttention, mean=simple average")

                                                                
                                                       
                                                                                                         
                                                  
    parser.add_argument("--view_drop", type=float, default=0.0,
                        help="view-level dropout prob for relation-wise regularization (0 disables)")
    parser.add_argument("--use_view_gate", action="store_true",
                        help="enable learnable view gates for each view channel")

                                     
    parser.add_argument("--hgnn_dim", type=int, default=None,
                        help="override hypergraph hidden dimension for sensitivity experiment")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning rate for sensitivity experiment")
    parser.add_argument("--exp_tag", type=str, default="default",
                        help="experiment tag used when saving sensitivity results")
    args, _ = parser.parse_known_args()
    THR_EVAL = "0.5"
    print(f"[THR_EVAL] mode={THR_EVAL} (val=thr from val best F1, 05=threshold 0.5, both=report both)")


    if args.ablation not in ABLATION_PRESETS:
        raise ValueError(f"[ABLATION] Unknown preset: {args.ablation}, choices={list(ABLATION_PRESETS.keys())}")
    ABLATION_PRESET = ABLATION_PRESETS[args.ablation]
                                                   
    if args.ablation == "wohe":
        args.dual_ablation = "node_only"
    if args.ablation == "wovw":
        args.view_fusion = "mean"

    ABLATION_MODE = str(args.ablation_mode).upper().strip()
    if ABLATION_MODE not in ("A", "B"):
        raise ValueError("[ABLATION] --ablation_mode must be A or B")

    apply_ablation_to_rec_weights(ABLATION_MODE, ABLATION_PRESET)
    print(f"[ABLATION] preset={args.ablation}, mode={ABLATION_MODE}, flags={ABLATION_PRESET}")
    print(f"[ABLATION] REC_W_CIRC_FINAL={REC_W_CIRC_FINAL}, REC_W_DIS_FINAL={REC_W_DIS_FINAL}")
                                                                    

                                    
    train_losses = []
    train_auc_list = []
    train_acc_list = []
    val_aupr_list = []
    fold_auc_list = []                  
    train_recall_list = []                            
                                            
    fold_summary = []                
               
    fold_test_auc = []
    fold_test_acc = []
    fold_best_aupr = []                                                
    fold_test_aupr = []                                   

                                                             
    val_true_folds, val_score_folds = [], []
    test_true_folds, test_score_folds = [], []

    best_thr_folds = []                     
    cm_folds = []                                     
    f1_test_folds = []                        
    mcc_test_folds = []                         

                                                  
    thr_grid = np.linspace(0.0, 1.0, 101)
    f1_grid_folds = []                   

                                   
    test_scores_pos_all = []
    test_scores_neg_all = []
                                                         
    acc_test_folds_05 = []               
    f1_test_folds_05 = []               
    mcc_test_folds_05 = []               

                                                                                    
    acc_test_folds_thr = []                  

                                     
                                   
                                     
    seed_base = 42
    deterministic = True

    if USE_SAVED_RUN_CONFIG and os.path.isfile(RUN_CONFIG_PATH):
        saved_cfg = load_json(RUN_CONFIG_PATH)

        seed_base = int(saved_cfg.get("seed_base", seed_base))
        deterministic = bool(saved_cfg.get("deterministic", deterministic))

        epochs = int(saved_cfg.get("epochs", 100))
        lr = float(saved_cfg.get("lr", 0.004))
        bio_out_dim = int(saved_cfg.get("bio_out_dim", 32))
        hgnn_dim = int(saved_cfg.get("hgnn_dim", 512))

        print(f"[REPRO] Loaded run config: {RUN_CONFIG_PATH}")
    else:
        epochs = 100
        lr = 0.004
        bio_out_dim = 32
        hgnn_dim = 512

                                          
    if args.hgnn_dim is not None:
        hgnn_dim = int(args.hgnn_dim)
        print(f"[SENSITIVITY] Override hgnn_dim from CLI: {hgnn_dim}")

    if args.lr is not None:
        lr = float(args.lr)
        print(f"[SENSITIVITY] Override lr from CLI: {lr}")

                       
    set_global_seed(seed_base, deterministic=deterministic)
    print(f"[REPRO] seed_base={seed_base}, deterministic={deterministic}")
                                                   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[REPRO] device initialized after seed:", device)

    CMI_data = './CD_data/'
                                     
                                                  
                                     
    if not USE_SAVED_RUN_CONFIG:
        run_id = make_run_id(seed_base)
        run_cfg = build_run_config(locals(), seed_base=seed_base, deterministic=deterministic, run_id=run_id)

                                                             
        save_json(run_cfg, RUN_CONFIG_PATH)

                         
        save_json(run_cfg, os.path.join(REPRO_DIR, f"{run_id}.json"))

        print(f"[REPRO] Saved run config to: {RUN_CONFIG_PATH}")
    else:
        print(f"[REPRO] Using saved run config: {RUN_CONFIG_PATH}")

                                  
    plot_data_collection = {}

    for fold in range(5):
        print('***********fold_{}*****************'.format(fold))
                                                  
        fold_seed = seed_base + fold
        set_global_seed(fold_seed, deterministic=deterministic)
        print(f"[REPRO] fold={fold}, fold_seed={fold_seed}")

                                                                   
        def _md5(path: str) -> str:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        files_to_check = [
            f'./data/disease_ids_unique_313.txt',
            f'./data/circBaseSequence_1502.csv',
            f'./data/{CMI_data}/Positive_Sample_Train{fold}.csv',
            f'./data/{CMI_data}/Positive_Sample_Validation{fold}.csv',
            f'./data/{CMI_data}/Positive_Sample_Test{fold}.csv',
            f'./data/{CMI_data}/Negative_Sample_Train{fold}.csv',
            f'./data/{CMI_data}/Negative_Sample_Validation{fold}.csv',
            f'./data/{CMI_data}/Negative_Sample_Test{fold}.csv',
            f'./data/./hyg_new/hyg_dict{fold}.pickle',
        ]
        for fp in files_to_check:
            abs_fp = os.path.abspath(fp)
            print(f"[DATA HASH] {abs_fp} | md5={_md5(abs_fp)}")

        fold_auc_single = []             

                                              
        global disease_to_idx, circ_to_idx
        disease_to_idx = {}
        ReadMyCsv4(disease_to_idx, 'data/disease_ids_unique_313.txt')            

        circ_to_idx = {}
        ReadMyCsv4(circ_to_idx, 'data/circBaseSequence_1502.csv')                

        n_circ = len(circ_to_idx)
        m_dis = len(disease_to_idx)
        print(f"[INFO] Nc={n_circ}, Nd={m_dis}")

                           
        PositiveSample_Train = []
        ReadMyCsv5(PositiveSample_Train, f'./data/{CMI_data}/Positive_Sample_Train{fold}.csv')
        PositiveSample_Validation = []
        ReadMyCsv5(PositiveSample_Validation, f'./data/{CMI_data}/Positive_Sample_Validation{fold}.csv')
        PositiveSample_Test = []
        ReadMyCsv5(PositiveSample_Test, f'./data/{CMI_data}/Positive_Sample_Test{fold}.csv')

        NegativeSample_Train = []
        ReadMyCsv5(NegativeSample_Train, f'./data/{CMI_data}/Negative_Sample_Train{fold}.csv')
        NegativeSample_Validation = []
        ReadMyCsv5(NegativeSample_Validation, f'./data/{CMI_data}/Negative_Sample_Validation{fold}.csv')
        NegativeSample_Test = []
        ReadMyCsv5(NegativeSample_Test, f'./data/{CMI_data}/Negative_Sample_Test{fold}.csv')

        x_train_pair = np.array(PositiveSample_Train + NegativeSample_Train)
        x_validation_pair = np.array(PositiveSample_Validation + NegativeSample_Validation)
        x_test_pair = np.array(PositiveSample_Test + NegativeSample_Test)
                                                         
                                                           
        y_train = MyLabel(x_train_pair)
        y_val = MyLabel(x_validation_pair)
        y_test = MyLabel(x_test_pair)

                                           
        circ_feat_matrix = np.load('./data/embedding/circRNA_embedding.npy')                       
        circ_feat_matrix = torch.from_numpy(circ_feat_matrix).float().to(device)

        disease_feat_matrix = np.load('./data/embedding/disease_embedding_313.npy')               
        disease_feat_matrix = torch.from_numpy(disease_feat_matrix).float().to(device)

                                          
        G_d, G_c, G_cc, G_dd, G_mc, G_md, S_cc, S_dd = \
            load_hyg(f'./data/hyg_new/hyg_dict{fold}.pickle')

                                                                   
        G_d, G_c, G_cc, G_dd, G_mc, G_md = apply_ablation_to_hyg(
            G_d, G_c, G_cc, G_dd, G_mc, G_md, ABLATION_PRESET
        )

                                
        hyg_edges = prepare_hyg_edges(G_d, G_c, G_cc, G_dd, G_mc, G_md, n_circ, m_dis, device)
        hypergraph_views = hyg_edges

        S_cc = S_cc.to(device)
        S_dd = S_dd.to(device)

                                                     
                                                           
        decode_dim = hgnn_dim // 4
        decoder = PairMLPDecoder(d=decode_dim, hidden=256, dropout=0.3).to(device)

                   
                                                    
        model = DHGCDA(
            BioEncoder(
                circ_feat_matrix.shape[1],
                disease_feat_matrix.shape[1],
                bio_out_dim
            ), HgnnEncoder(
                bio_out_dim, hgnn_dim, dual_ablation=args.dual_ablation,
    view_fusion=args.view_fusion
            ), decoder).to(device)
                         

        loss_function1 = torch.nn.BCEWithLogitsLoss().to(device)
        loss_function2 = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                                                        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=6,
            threshold=1e-4,
            min_lr=1e-5,
            verbose=False
        )

                                                                                 
        from sklearn.metrics import average_precision_score

                            
        best_aupr = -1.0
        best_auc = -1.0
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        patience = 15                                     
        no_improve = 0
        epochs_run = 0                                  

        for e in range(epochs):
            train(
                circ_feat_matrix,
                disease_feat_matrix,
                x_train_pair,
                np.array(y_train),
                hypergraph_views,
                S_cc,
                S_dd,
                e
            )

                                                                            
            ModelValOutput = test(
                circ_feat_matrix,
                disease_feat_matrix,
                x_validation_pair,
                np.array(y_val),
                hypergraph_views
            )

            aupr_val = average_precision_score(np.array(y_val), ModelValOutput)
            val_aupr_list.append(aupr_val)
            fold_auc_single.append(train_auc_list[-1])                      
            epochs_run = e + 1

                                                     
            auc_val = metrics.roc_auc_score(np.array(y_val), ModelValOutput)
            scheduler.step(auc_val)

                                                        
            if auc_val > best_auc + 1e-4:
                best_auc = auc_val
                best_aupr = aupr_val                                
                best_epoch = e
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"[EarlyStopping] fold {fold}, epoch {e+1}: "
                        f"val AUC 连续 {patience} 个 epoch 未提升，"
                        f"最佳 epoch = {best_epoch+1}, best val AUC = {best_auc:.4f}"
                    )
                    break
                                                          
        if epochs_run == 0:
            epochs_run = epochs

                                                  
                                            
        if epochs_run < epochs:
            repeat = epochs - epochs_run
            last_loss = train_losses[-1]
            last_auc = train_auc_list[-1]
            last_acc = train_acc_list[-1]
            last_aupr = val_aupr_list[-1]
            last_fold_auc = fold_auc_single[-1]

            train_losses.extend([last_loss] * repeat)
            train_auc_list.extend([last_auc] * repeat)
            train_acc_list.extend([last_acc] * repeat)
            val_aupr_list.extend([last_aupr] * repeat)
            fold_auc_single.extend([last_fold_auc] * repeat)
                                                                      
            last_recall = train_recall_list[-1] if len(train_recall_list) > 0 else 0.0
            train_recall_list.extend([last_recall] * (epochs - epochs_run))

                                     
        model.load_state_dict(best_state)

                                                                  
        val_logits = test_logits(
            circ_feat_matrix,
            disease_feat_matrix,
            x_validation_pair,
            hypergraph_views
        )
        val_labels = torch.from_numpy(np.array(y_val)).float().to(device)

        if USE_TEMPERATURE_SCALING:
            scaler = TemperatureScaler().to(device)
            optimizer_t = torch.optim.LBFGS([scaler.temperature], lr=0.1, max_iter=50)


            def eval_t():
                optimizer_t.zero_grad()
                loss_t = torch.nn.functional.binary_cross_entropy_with_logits(
                    scaler(val_logits), val_labels
                )
                loss_t.backward()
                return loss_t


            optimizer_t.step(eval_t)
            T_opt = float(scaler.temperature.item())
        else:
                                       
            scaler = torch.nn.Identity().to(device)
            T_opt = 1.0

        print(
            f"[Info] fold {fold}: 使用最佳 epoch = {best_epoch+1}, "
            f"AUPR = {best_aupr:.4f}, best val AUC = {best_auc:.4f} 作为最终模型进行测试"
        )
        fold_best_aupr.append(best_aupr)

                                                                                  










        with torch.no_grad():
            val_scores_best = torch.sigmoid(
                scaler(val_logits)
            ).cpu().numpy()

        y_val_np = np.array(y_val).astype(int)
        val_scores_best = np.array(val_scores_best)

                              
        prec_v, rec_v, thr_v = metrics.precision_recall_curve(y_val_np, val_scores_best)
        eps = 1e-12
        f1_v = 2 * prec_v[:-1] * rec_v[:-1] / (prec_v[:-1] + rec_v[:-1] + eps)            
        best_idx = int(np.argmax(f1_v))
        best_thr = float(thr_v[best_idx])

        best_thr_folds.append(best_thr)
        val_true_folds.append(y_val_np)
        val_score_folds.append(val_scores_best)

                                         
        f1_grid = []
        for t in thr_grid:
            pred = (val_scores_best >= t).astype(int)
            f1_grid.append(metrics.f1_score(y_val_np, pred, zero_division=0))
        f1_grid_folds.append(np.array(f1_grid))

                                          
                                                                
        test_logits_fold = test_logits(
            circ_feat_matrix,
            disease_feat_matrix,
            x_test_pair,
            hypergraph_views
        )

        with torch.no_grad():
            y_score_fold = torch.sigmoid(
                scaler(test_logits_fold)
            ).cpu().numpy()

        y_true_fold = np.array(y_test)

                                                                         
        y_true_fold = y_true_fold.astype(int)
        test_true_folds.append(y_true_fold)
        test_score_folds.append(y_score_fold)
                                            
        aupr_test_fold = metrics.average_precision_score(y_true_fold, y_score_fold)
        fold_test_aupr.append(float(aupr_test_fold))

                     
        test_scores_pos_all.append(y_score_fold[y_true_fold == 1])
        test_scores_neg_all.append(y_score_fold[y_true_fold == 0])

                                                                       
        thr_use = best_thr_folds[-1]
        y_pred_thr = (y_score_fold >= thr_use).astype(int)
        y_pred_05 = (y_score_fold > 0.5).astype(int)

                    
        acc_thr = metrics.accuracy_score(y_true_fold, y_pred_thr)               
        acc_05 = metrics.accuracy_score(y_true_fold, y_pred_05)           

                         
        f1_thr = metrics.f1_score(y_true_fold, y_pred_thr, zero_division=0)
        mcc_thr = metrics.matthews_corrcoef(y_true_fold, y_pred_thr) if len(np.unique(y_true_fold)) > 1 else 0.0

        f1_05 = metrics.f1_score(y_true_fold, y_pred_05, zero_division=0)
        mcc_05 = metrics.matthews_corrcoef(y_true_fold, y_pred_05) if len(np.unique(y_true_fold)) > 1 else 0.0

                                            
        cm = metrics.confusion_matrix(y_true_fold, y_pred_thr, labels=[0, 1])
        cm_folds.append(cm)

                               
        acc_test_folds_thr.append(float(acc_thr))
        acc_test_folds_05.append(float(acc_05))

        f1_test_folds.append(float(f1_thr))                      
        mcc_test_folds.append(float(mcc_thr))

        f1_test_folds_05.append(float(f1_05))
        mcc_test_folds_05.append(float(mcc_05))

                                    
        if THR_EVAL == "val":
            print(
                f"[Fold {fold + 1}] ACC@thr_val={acc_thr:.4f} | F1@thr_val={f1_thr:.4f} | MCC@thr_val={mcc_thr:.4f} | thr={thr_use:.3f}")
        elif THR_EVAL == "05":
            print(f"[Fold {fold + 1}] ACC@0.5={acc_05:.4f} | F1@0.5={f1_05:.4f} | MCC@0.5={mcc_05:.4f}")
        else:
            print(f"[Fold {fold + 1}] ACC@0.5={acc_05:.4f} | ACC@thr_val={acc_thr:.4f} | "
                  f"F1@0.5={f1_05:.4f} | F1@thr_val={f1_thr:.4f} | "
                  f"MCC@0.5={mcc_05:.4f} | MCC@thr_val={mcc_thr:.4f} | thr={thr_use:.3f}")

                                                        
        auc_final_fold = metrics.roc_auc_score(y_true_fold, y_score_fold)

                                    
                                         
        slice_start = fold * epochs
        slice_end = (fold + 1) * epochs

        fold_train_loss = train_losses[slice_start:slice_end]
        fold_train_auc = train_auc_list[slice_start:slice_end]
        fold_train_acc = train_acc_list[slice_start:slice_end]
        fold_val_aupr = val_aupr_list[slice_start:slice_end]

                        
        plot_data_collection[fold] = {
                                        
            'train_loss': fold_train_loss,
            'train_auc': fold_train_auc,
            'train_acc': fold_train_acc,
            'val_aupr': fold_val_aupr,

                                               
            'test_y_true': y_true_fold,              
            'test_y_score': y_score_fold,              
            'test_auc': auc_final_fold,                   

                                       
            'val_y_true_best': y_val_np,
            'val_y_score_best': val_scores_best,
            'best_thr': best_thr,           

                           
            'best_epoch': best_epoch + 1,
            'best_val_auc': best_auc,
            'test_aupr': aupr_test_fold
        }
                                      
        print(f"[INFO] Fold {fold} 画图数据已收集 (Test AUC: {auc_final_fold:.4f})")

                                                                    
        f1_t = f1_thr
        mcc_t = mcc_thr

                                         
        y_pred_label_fold = (y_score_fold > 0.5).astype(int)
                      
        acc_final_fold = metrics.accuracy_score(y_true_fold, y_pred_label_fold)
                      
                                                                          

        fold_test_auc.append(auc_final_fold)
        fold_test_acc.append(acc_final_fold)

                                              
        fold_summary.append({
            "fold": fold + 1,
            "AUC": float(auc_final_fold),
            "AUPR": float(aupr_test_fold),

                        
            "ACC@0.5": float(acc_05),
            "F1@0.5": float(f1_05),
            "MCC@0.5": float(mcc_05),

            "ACC@thr_val": float(acc_thr),
            "F1@thr_val": float(f1_thr),
            "MCC@thr_val": float(mcc_thr),

            "thr": float(best_thr),
        })

        fold_auc_list.append(fold_auc_single)

                        
    mean_auc = np.mean(fold_test_auc)
    std_auc = np.std(fold_test_auc)
    mean_acc = np.mean(fold_test_acc)
    std_acc = np.std(fold_test_acc)
    mean_aupr = np.mean(fold_best_aupr)                         
    std_aupr = np.std(fold_best_aupr)
                                                            
    mean_test_aupr = np.mean(fold_test_aupr)
    std_test_aupr = np.std(fold_test_aupr)

    mean_f1 = np.mean(f1_test_folds)
    std_f1 = np.std(f1_test_folds)
    mean_mcc = np.mean(mcc_test_folds)
    std_mcc = np.std(mcc_test_folds)
    mean_thr = np.mean(best_thr_folds)
    std_thr = np.std(best_thr_folds)
                                                
    mean_acc_05 = np.mean(acc_test_folds_05); std_acc_05 = np.std(acc_test_folds_05)
    mean_f1_05  = np.mean(f1_test_folds_05);  std_f1_05  = np.std(f1_test_folds_05)
    mean_mcc_05 = np.mean(mcc_test_folds_05); std_mcc_05 = np.std(mcc_test_folds_05)

    mean_acc_thr = np.mean(acc_test_folds_thr); std_acc_thr = np.std(acc_test_folds_thr)
    mean_f1_thr  = np.mean(f1_test_folds);      std_f1_thr  = np.std(f1_test_folds)
    mean_mcc_thr = np.mean(mcc_test_folds);     std_mcc_thr = np.std(mcc_test_folds)

















    print("当前时间:", now.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n========== 最终 5-Fold 结果 ==========")
    print("AUC_test: {:.6f} ± {:.6f}".format(mean_auc, std_auc))
    print("ACC_test: {:.6f} ± {:.6f}".format(mean_acc_05, std_acc_05))
    print("AUPR_test: {:.6f} ± {:.6f}".format(mean_test_aupr, std_test_aupr))
    print("F1_test : {:.6f} ± {:.6f}".format(mean_f1_05, std_f1_05))
    print("MCC_test: {:.6f} ± {:.6f}".format(mean_mcc_05, std_mcc_05))
    print("thr_val : {:.6f} ± {:.6f}".format(mean_thr, std_thr))

                               
    print("AUPR_val(best): {:.6f} ± {:.6f}".format(mean_aupr, std_aupr))

    end_time = time.time()
    print("总运行时间：{:.2f} 秒".format(end_time - start_time))


                                   
    num_folds = 5

                
    auc_folds = np.array(fold_auc_list)                   
    loss_folds = split_by_fold(train_losses, num_folds, epochs)
    acc_folds = split_by_fold(train_acc_list, num_folds, epochs)
    recall_folds = split_by_fold(train_recall_list, num_folds, epochs)
    aupr_folds = split_by_fold(val_aupr_list, num_folds, epochs)

                                            
    train_auc_last = auc_folds[:, -1]                      
    train_acc_last = acc_folds[:, -1]
    train_recall_last = recall_folds[:, -1]
    train_loss_last = loss_folds[:, -1]

    print("Train_AUC(final):    {:.6f} ± {:.6f}".format(
        train_auc_last.mean(), train_auc_last.std()))
    print("Train_ACC(final):    {:.6f} ± {:.6f}".format(
        train_acc_last.mean(), train_acc_last.std()))
    print("Train_Recall(final): {:.6f} ± {:.6f}".format(
        train_recall_last.mean(), train_recall_last.std()))
    print("Train_Loss(final):   {:.6f} ± {:.6f}".format(
        train_loss_last.mean(), train_loss_last.std()))
    print("=====================================\n")

                            
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))          
    axes = axes.flatten()              

             
    plot_metric_mean_std(
        auc_folds,
        title="Training AUC (5-fold mean ± std)",
        ylabel="AUC",
        metric_name="AUC",
        ax=axes[0]
    )

    plot_metric_mean_std(
        recall_folds,
        title="Training Recall (5-fold mean ± std)",
        ylabel="Recall",
        metric_name="Recall",
        ax=axes[1]
    )

    plot_metric_mean_std(
        acc_folds,
        title="Training ACC (5-fold mean ± std)",
        ylabel="Accuracy",
        metric_name="ACC",
        ax=axes[2]
    )

    plot_metric_mean_std(
        loss_folds,
        title="Training Loss (5-fold mean ± std)",
        ylabel="Loss",
        metric_name="Loss",
        ax=axes[3]
    )

    plot_metric_mean_std(
        aupr_folds,
        title="Validation AUPR (5-fold mean ± std)",
        ylabel="AUPR",
        metric_name="AUPR",
        ax=axes[4]
    )

                        
    folds = np.arange(1, num_folds + 1)
    width = 0.35
    ax = axes[5]
    ax.bar(folds - width / 2, fold_test_auc, width, label="AUC_test")
    ax.bar(folds + width / 2, fold_test_acc, width, label="ACC_test")
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("5-Fold Test Performance")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()

            
    plt.tight_layout()

    out_path = "A.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
              
    plt.show()
                                                                  
                                                                      
                                                                  
    import os

                 
    pos_all = np.concatenate(test_scores_pos_all) if len(test_scores_pos_all) else np.array([])
    neg_all = np.concatenate(test_scores_neg_all) if len(test_scores_neg_all) else np.array([])

                                    
    fpr_grid = np.linspace(0.0, 1.0, 201)
    tpr_folds = []
    auc_each = []
    for yt, ys in zip(test_true_folds, test_score_folds):
        fpr, tpr, _ = metrics.roc_curve(yt, ys)
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tpr_folds.append(tpr_i)
        auc_each.append(metrics.roc_auc_score(yt, ys))
    tpr_folds = np.array(tpr_folds)
    tpr_mean = tpr_folds.mean(axis=0)
    tpr_std = tpr_folds.std(axis=0)

                                      
    rec_grid = np.linspace(0.0, 1.0, 201)
    pre_folds = []
    ap_each = []
    for yt, ys in zip(test_true_folds, test_score_folds):
        pre, rec, _ = metrics.precision_recall_curve(yt, ys)
                     
        order = np.argsort(rec)
        rec_s = rec[order]
        pre_s = pre[order]
        pre_i = np.interp(rec_grid, rec_s, pre_s)
        pre_folds.append(pre_i)
        ap_each.append(metrics.average_precision_score(yt, ys))
    pre_folds = np.array(pre_folds)
    pre_mean = pre_folds.mean(axis=0)
    pre_std = pre_folds.std(axis=0)

                                      
    f1_grid_folds = np.array(f1_grid_folds)            
    f1_thr_mean = f1_grid_folds.mean(axis=0)
    f1_thr_std = f1_grid_folds.std(axis=0)
    best_thr_mean = float(np.mean(best_thr_folds))

                                                 
    cm_mean = np.mean(np.array(cm_folds), axis=0)
    cm_row_sum = cm_mean.sum(axis=1, keepdims=True) + 1e-12
    cm_norm = cm_mean / cm_row_sum                        

                
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Extra Evaluation Plots (5-fold)", fontsize=14)

               
    ax = axes[0, 0]
    for i, (yt, ys) in enumerate(zip(test_true_folds, test_score_folds), start=1):
        fpr, tpr, _ = metrics.roc_curve(yt, ys)
        auc_i = metrics.roc_auc_score(yt, ys)
        ax.plot(
            fpr, tpr,
            linewidth=1.4,
            alpha=0.95,
            linestyle="-",
            label=f"ROC fold {i} (AUC = {auc_i:.4f})"
        )

    roc_label = f"Mean ROC (AUC = {np.mean(auc_each):.4f})"
    ax.plot(
        fpr_grid, tpr_mean,
        linewidth=2.2,
        linestyle="-",
        label=roc_label
    )
    ax.fill_between(
        fpr_grid,
        np.clip(tpr_mean - tpr_std, 0, 1),
        np.clip(tpr_mean + tpr_std, 0, 1),
        alpha=0.15,
        label="_nolegend_"
    )
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.0,
        color="gray",
        alpha=0.8,
        label="_nolegend_"
    )
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

              
    ax = axes[0, 1]

    all_test_true = np.concatenate([np.asarray(y).reshape(-1) for y in test_true_folds])
    pos_rate = float(np.mean(all_test_true))

    for i, (yt, ys) in enumerate(zip(test_true_folds, test_score_folds), start=1):
        pre, rec, _ = metrics.precision_recall_curve(yt, ys)
        ap_i = metrics.average_precision_score(yt, ys)
        ax.plot(
            rec, pre,
            linewidth=1.4,
            alpha=0.95,
            linestyle="-",
            label=f"PR fold {i} (AUPR = {ap_i:.4f})"
        )

    pr_label = f"Mean PR (AUPR = {np.mean(ap_each):.4f})"
    ax.plot(
        rec_grid, pre_mean,
        linewidth=2.2,
        linestyle="-",
        label=pr_label
    )
    ax.fill_between(
        rec_grid,
        np.clip(pre_mean - pre_std, 0, 1),
        np.clip(pre_mean + pre_std, 0, 1),
        alpha=0.15,
        label="_nolegend_"
    )
    ax.plot(
        [0, 1], [1, 0],
        linestyle="--",
        linewidth=1.0,
        color="gray",
        alpha=0.8,
        label="_nolegend_"
    )
    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower left", frameon=True)



                             
    ax = axes[0, 2]
    for i, f1g in enumerate(f1_grid_folds):
        ax.plot(thr_grid, f1g, linewidth=1, alpha=0.25)
    ax.plot(thr_grid, f1_thr_mean, linewidth=2, label="Mean F1(thr) on Val")
    ax.fill_between(thr_grid, np.clip(f1_thr_mean - f1_thr_std, 0, 1), np.clip(f1_thr_mean + f1_thr_std, 0, 1),
                    alpha=0.2, label="±1 std")
    ax.axvline(best_thr_mean, linestyle="--", linewidth=1.5, alpha=0.8, label=f"Mean best thr={best_thr_mean:.3f}")
    ax.set_title("F1 vs Threshold (Validation)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

                                           
    ax = axes[1, 0]
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)
    ax.set_title("Confusion Matrix (Test, thr from Val)\nRow-normalized mean over folds")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]);
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]);
    ax.set_yticklabels(["0", "1"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm_norm[r, c] * 100:.1f}%", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                                    
    ax = axes[1, 1]
    ax.hist(neg_all, bins=30, alpha=0.5, label="Neg (test)")
    ax.hist(pos_all, bins=30, alpha=0.5, label="Pos (test)")
    ax.set_title("Score Distribution (Test, all folds)")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

                                             
    ax = axes[1, 2]
    folds = np.arange(1, len(f1_test_folds) + 1)
    w = 0.35
    ax.bar(folds - w / 2, f1_test_folds, w, label="F1_test (thr from Val)")
    ax.bar(folds + w / 2, mcc_test_folds, w, label="MCC_test (thr from Val)")
    ax.set_title("Fold-wise Thresholded Metrics (Test)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_xticks(folds)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                
    out_path = "B"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[DONE] Extra figure saved to: {os.path.abspath(out_path)}")

                                                                  
                            
                                                                  
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    timestamp = time.strftime("%Y%m%d_%H%M%S")

                                                               
    fig_roc, ax = plt.subplots(figsize=(5.2, 4.6))

    for i, (yt, ys) in enumerate(zip(test_true_folds, test_score_folds), start=1):
        fpr, tpr, _ = metrics.roc_curve(yt, ys)
        auc_i = metrics.roc_auc_score(yt, ys)
        ax.plot(
            fpr, tpr,
            linewidth=1.4,
            alpha=0.95,
            linestyle="-",
            label=f"ROC fold {i} (AUC = {auc_i:.4f})"
        )

    roc_label = f"Mean ROC (AUC = {np.mean(auc_each):.4f})"
    ax.plot(
        fpr_grid, tpr_mean,
        linewidth=2.2,
        linestyle="-",
        label=roc_label
    )

    ax.fill_between(
        fpr_grid,
        np.clip(tpr_mean - tpr_std, 0, 1),
        np.clip(tpr_mean + tpr_std, 0, 1),
        alpha=0.18,
        label="_nolegend_"
    )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        alpha=0.8,
        label="_nolegend_"
    )

    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    roc_pdf_path = f"ROC_curve_{timestamp}.pdf"
    fig_roc.tight_layout()
    fig_roc.savefig(roc_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig_roc)

    print(f"[DONE] ROC PDF saved to: {os.path.abspath(roc_pdf_path)}")

                                                              
    fig_pr, ax = plt.subplots(figsize=(5.2, 4.6))

    all_test_true = np.concatenate([np.asarray(y).reshape(-1) for y in test_true_folds])
    pos_rate = float(np.mean(all_test_true))

    for i, (yt, ys) in enumerate(zip(test_true_folds, test_score_folds), start=1):
        pre, rec, _ = metrics.precision_recall_curve(yt, ys)
        ap_i = metrics.average_precision_score(yt, ys)
        ax.plot(
            rec, pre,
            linewidth=1.4,
            alpha=0.95,
            linestyle="-",
            label=f"PR fold {i} (AUPR = {ap_i:.4f})"
        )

    pr_label = f"Mean PR (AUPR = {np.mean(ap_each):.4f})"
    ax.plot(
        rec_grid, pre_mean,
        linewidth=2.2,
        linestyle="-",
        label=pr_label
    )

    ax.fill_between(
        rec_grid,
        np.clip(pre_mean - pre_std, 0, 1),
        np.clip(pre_mean + pre_std, 0, 1),
        alpha=0.18,
        label="_nolegend_"
    )

    ax.plot(
        [0, 1], [1, 0],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        alpha=0.8,
        label="_nolegend_"
    )

    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    pr_pdf_path = f"PR_curve_{timestamp}.pdf"
    fig_pr.tight_layout()
    fig_pr.savefig(pr_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig_pr)

    print(f"[DONE] PR PDF saved to: {os.path.abspath(pr_pdf_path)}")

    print("\n========== Fold-wise Detailed Results ==========")
    for item in fold_summary:
        print(
            f"Fold {item['fold']}: "
            f"AUC={item['AUC']:.4f}, "
            f"AUPR={item['AUPR']:.4f}, "
            f"ACC={item['ACC@0.5']:.4f}, "
            f"F1={item['F1@0.5']:.4f}, "
            f"MCC={item['MCC@0.5']:.4f}, "
                                                       
                                                     
                                                       
            f"thr={item['thr']:.4f}"
        )
    print("===============================================\n")

                                         
    mean_auc = float(np.mean(fold_test_auc))
    std_auc = float(np.std(fold_test_auc))
    mean_aupr = float(np.mean(fold_test_aupr))
    std_aupr = float(np.std(fold_test_aupr))
    mean_acc = float(np.mean(fold_test_acc)) if len(fold_test_acc) > 0 else 0.0
    mean_f1 = float(np.mean(f1_test_folds_05)) if len(f1_test_folds_05) > 0 else 0.0
    mean_mcc = float(np.mean(mcc_test_folds_05)) if len(mcc_test_folds_05) > 0 else 0.0

                                         
    sens_name = None
    sens_value = None

    if args.hgnn_dim is not None:
        sens_name = "hgnn_dim"
        sens_value = hgnn_dim
    elif args.lr is not None:
        sens_name = "lr"
        sens_value = lr

    if sens_name is not None:
        print(f"[SUMMARY] {sens_name}={sens_value}, AUC={mean_auc:.4f}±{std_auc:.4f}, "
              f"AUPR={mean_aupr:.4f}±{std_aupr:.4f}")

        sens_csv_path = os.path.join(REPRO_DIR, f"{sens_name}_sensitivity.csv")
        os.makedirs(REPRO_DIR, exist_ok=True)

        write_header = not os.path.exists(sens_csv_path)
        with open(sens_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "exp_tag", sens_name,
                    "mean_auc", "std_auc",
                    "mean_aupr", "std_aupr",
                    "mean_acc", "mean_f1", "mean_mcc"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.exp_tag,
                sens_value,
                f"{mean_auc:.6f}",
                f"{std_auc:.6f}",
                f"{mean_aupr:.6f}",
                f"{std_aupr:.6f}",
                f"{mean_acc:.6f}",
                f"{mean_f1:.6f}",
                f"{mean_mcc:.6f}",
            ])

        print(f"[DONE] Sensitivity result appended to: {sens_csv_path}")

                                      
                                    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"plot_AUC_{np.mean(fold_test_auc):.4f}_data_{timestamp}.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(plot_data_collection, f)

    print(f"\n[SUCCESS] 所有画图数据已保存至: {save_path}")
    print("您可以直接读取该文件来绘制 Loss曲线、ROC曲线、PR曲线等。")
