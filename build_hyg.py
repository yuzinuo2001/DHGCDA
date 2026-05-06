



























import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import csv
import torch
csv.field_size_limit(2**31 - 1)
                                        
        
                                        

def build_hyperedge_index_from_bipartite(node_num, hyperedge_num, incidence_list):




    if len(incidence_list) == 0:
                
        return np.zeros((2, 0), dtype=np.int64)
    edge_index = np.array(incidence_list, dtype=np.int64).T          
    return edge_index


def build_neighbor_list(num_nodes, adj_dict, max_deg=None):






    if max_deg is None:
        max_deg = max((len(v) for v in adj_dict.values()), default=1)

    hye_len = np.zeros(num_nodes, dtype=np.float32)
    hye_node = np.zeros((num_nodes, max_deg), dtype=np.int64)                

    for i in range(num_nodes):
        neigh = sorted(list(adj_dict.get(i, [])))
        hye_len[i] = len(neigh)
        for j, n in enumerate(neigh[:max_deg]):
            hye_node[i, j] = n + 1                                          

    return hye_len, hye_node


def build_topk_hypergraph_from_similarity(sim_mat, top_n=5):





    N = sim_mat.shape[0]
    incidence = []
    adj_dict = defaultdict(set)

    for i in range(N):
                      
        idx_sorted = np.argsort(sim_mat[i])      
        idx_top = idx_sorted[-(top_n + 1):]         
        nbrs = [j for j in idx_top if j != i][:top_n]

        for n in nbrs:
            incidence.append((n, i))                        
            adj_dict[i].add(n)

    edge_index = build_hyperedge_index_from_bipartite(N, N, incidence)
    hye_len, hye_node = build_neighbor_list(N, adj_dict, max_deg=top_n)
    return edge_index, hye_len, hye_node


                                        
                               
                                        

def load_circ_ids(circ_ids_path="circBaseSequence_1502.csv"):


    circRNAs = []
    with open(circ_ids_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cid = str(row[0]).strip()
            if cid:
                circRNAs.append(cid)
                   
    circRNAs = list(dict.fromkeys(circRNAs))
    circ_to_idx = {cid: i for i, cid in enumerate(circRNAs)}
    print(f"[INFO] circ_ids 数量: {len(circRNAs)} (来自 {circ_ids_path})")
    return circRNAs, circ_to_idx

                                        
                                                          
                                        

def load_disease_ids(dis_ids_path="disease_ids_unique_313.txt"):


    diseases = []
    with open(dis_ids_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = str(row[0]).strip()
            if name:
                diseases.append(name)
             
    diseases = list(dict.fromkeys(diseases))
    disease_to_idx = {d: i for i, d in enumerate(diseases)}
    print(f"[INFO] 从 {dis_ids_path} 读取疾病列表 Nd = {len(diseases)}")
    return diseases, disease_to_idx


                                        
                            
                                        

def load_circ_disease_entries(path="The circRNA-disease entries.xlsx"):
                     
    df = pd.read_excel(path)
               
    df = df[df["Species"] == "Human"].copy()

    df["circ_id"] = df["circRNA Name"].astype(str).str.strip()
    df["disease"] = df["Disease Name"].astype(str).str.strip()

                          
    df = df.dropna(subset=["circ_id", "disease"])
    print(f"[INFO] Human 物种条目数: {len(df)}")
    return df


                                        
                                                        
                                        

def filter_by_circ_ids(df, circ_to_idx, disease_to_idx):




    before = len(df)
                                   
    df = df[df["circ_id"].isin(circ_to_idx.keys())].copy()
    after = len(df)
    print(f"[INFO] circRNA 有 embedding 的条目: {after} / {before}")

                                            
    df = df[df["disease"].isin(disease_to_idx.keys())].copy()

                                   
    dis_in_data = sorted(df["disease"].unique().tolist())
    print(f"[INFO] 关联表中实际出现的疾病种类数: {len(dis_in_data)} / {len(disease_to_idx)}")

    return df



                                        
                                        
                                        

def build_pairs(df, circ_to_idx, disease_to_idx):
    pair_idx = []
    for _, row in df.iterrows():
        c = circ_to_idx[row["circ_id"]]
        d = disease_to_idx[row["disease"]]
        pair_idx.append((c, d))
    pair_idx = list(set(pair_idx))      
    print(f"[INFO] 唯一 circ–disease 关联数: {len(pair_idx)}")
    return pair_idx

            
def load_circ_mirna_edges(cmi_files, circ_to_idx):




    import pandas as pd
    from collections import defaultdict

    mir2circs = defaultdict(set)

    for f in cmi_files:
        df = pd.read_csv(f)
                                                  
        df = df.rename(columns={df.columns[0]: "miRNA", df.columns[1]: "circ"})
        df["miRNA"] = df["miRNA"].astype(str).str.strip()
        df["circ"] = df["circ"].astype(str).str.strip()

                       
        df = df[df["circ"].isin(circ_to_idx.keys())]

        for _, row in df.iterrows():
            mir = row["miRNA"]
            c = circ_to_idx[row["circ"]]
            mir2circs[mir].add(c)

    print(f"[INFO] miRNA 超边数(去重后) = {len(mir2circs)}")
    return mir2circs


def build_hy_mi(n_circ, mir2circs, max_deg_mi=50, min_deg_mi=2):





    import numpy as np
    import torch

                      
    mir_items = [(m, sorted(list(cs))) for m, cs in mir2circs.items() if len(cs) >= min_deg_mi]
    mir_items = sorted(mir_items, key=lambda x: x[0])                     

    n_miRNA = len(mir_items)
    if n_miRNA == 0:
              
        G_mc = [
            torch.from_numpy(np.zeros((2, 0), dtype=np.int64)).long(),
            torch.from_numpy(np.zeros((1,), dtype=np.float32)).float(),
            torch.from_numpy(np.zeros((1, 1), dtype=np.int64)).long()
        ]
        miRNA_ids = []
        print("[WARN] hy_mi 为空：没有满足 min_deg_mi 的 miRNA 超边")
        return G_mc, miRNA_ids

    incidence = []
    hye_len = np.zeros(n_miRNA, dtype=np.float32)
    hye_node = np.zeros((n_miRNA, max_deg_mi), dtype=np.int64)             

    for h_idx, (mir, circs) in enumerate(mir_items):
                                           
        circs = circs[:max_deg_mi]
        hye_len[h_idx] = len(circs)
        for j, c_idx in enumerate(circs):
            incidence.append((c_idx, h_idx))                               
            hye_node[h_idx, j] = c_idx + 1                              

    edge_index = build_hyperedge_index_from_bipartite(n_circ, n_miRNA, incidence)

    G_mc = [
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(hye_len).float(),
        torch.from_numpy(hye_node).long()
    ]
    miRNA_ids = [m for m, _ in mir_items]
    print(f"[INFO] hy_mi: Nc={n_circ}, Nmi={n_miRNA}, max_deg_mi={max_deg_mi}, min_deg_mi={min_deg_mi}")
    return G_mc, miRNA_ids


def _norm_name_casefold(s: str) -> str:

    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s.casefold()


def build_hy_do_from_mirna_disease_csv(
        disease_ids_file: str,
        md_csv_file: str,
        max_deg_do: int = 80,
        min_deg_do: int = 2,
):







    import difflib
    import re

                                   
    def _norm(s):
        if not isinstance(s, str): return ""
                                                                            
        s = s.lower()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

                                          
                                              
    diseases, _ = load_disease_ids(disease_ids_file)

                                              
                                                      
    target_norm_map = {}
    for i, name in enumerate(diseases):
        norm_name = _norm(name)
        if norm_name:
            target_norm_map[norm_name] = i

    target_norms = list(target_norm_map.keys())

                 
                             
    try:
        df = pd.read_csv(md_csv_file, encoding="utf-8")
    except:
        df = pd.read_csv(md_csv_file, encoding="latin1")

    if "miRNA" not in df.columns or "disease" not in df.columns:
        raise ValueError(f"[G_md] CSV must contain columns: miRNA, disease. Got columns={list(df.columns)}")

                                                                        
    unique_sources = df["disease"].astype(str).unique()
    source_to_idx = {}

    print(f"[INFO] 开始构建疾病名称模糊匹配映射 (Source: {len(unique_sources)} -> Target: {len(diseases)})...")

    matched_count = 0
    for src in unique_sources:
        src_norm = _norm(src)
        if not src_norm: continue

                                              
        if src_norm in target_norm_map:
            source_to_idx[src] = target_norm_map[src_norm]
            matched_count += 1
            continue

                                                                       
        src_tokens = sorted(src_norm.split())
        found_token = False
        for tgt in target_norms:
            if sorted(tgt.split()) == src_tokens:
                source_to_idx[src] = target_norm_map[tgt]
                found_token = True
                break
        if found_token:
            matched_count += 1
            continue

                                             
                                                                            
                                                                                
                                                                                            
        candidates = []
        for tgt in target_norms:
                                                           
            if len(tgt) > 3 and tgt in src_norm:
                candidates.append(tgt)

        if candidates:
            best = max(candidates, key=len)
            source_to_idx[src] = target_norm_map[best]
            matched_count += 1
            continue

                                                           
                                                                               
        close = difflib.get_close_matches(src_norm, target_norms, n=1, cutoff=0.85)
        if close:
            source_to_idx[src] = target_norm_map[close[0]]
            matched_count += 1
            continue

    print(f"[INFO] 模糊匹配完成。匹配成功主要疾病数: {matched_count} / {len(unique_sources)}")

                         
    mir2dis = defaultdict(set)
    total_rows = 0
    hit_rows = 0

    for mir, dis_name in zip(df["miRNA"].astype(str), df["disease"].astype(str)):
        total_rows += 1
        mir = mir.strip()

                                  
        if dis_name in source_to_idx:
            d_idx = source_to_idx[dis_name]
            mir2dis[mir].add(d_idx)
            hit_rows += 1

                      
    hyperedges = []
    for mir, dis_set in mir2dis.items():
        if len(dis_set) >= min_deg_do:
            dis_list = sorted(list(dis_set))
            if len(dis_list) > max_deg_do:
                dis_list = dis_list[:max_deg_do]
            hyperedges.append((mir, dis_list))

    m_dis = len(diseases)
    E = len(hyperedges)

    row, col = [], []
    hye_len, hye_node = [], []

    for e_idx, (mir, dis_list) in enumerate(hyperedges):
        hye_len.append(len(dis_list))

                                                               
        node_1based = [d + 1 for d in dis_list]
        if len(node_1based) < max_deg_do:
            node_1based = node_1based + [0] * (max_deg_do - len(node_1based))
        hye_node.append(node_1based)

        for d in dis_list:
            row.append(d)
            col.append(e_idx)

    if len(row) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        hye_len = torch.zeros((0,), dtype=torch.long)
        hye_node = torch.zeros((0, max_deg_do), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        hye_len = torch.tensor(hye_len, dtype=torch.long)
        hye_node = torch.tensor(hye_node, dtype=torch.long)

    print(f"[hy_do <- G_md] Nd={m_dis}, NmiRNA={E}, nnz={edge_index.size(1)}")
    print(f"[hy_do <- G_md] matched_rows={hit_rows}/{total_rows} (enhanced matching)")

    G_md = [edge_index, hye_len, hye_node]
    return G_md

                                        
                     
                                        

def build_hy_mc_cm(n_circ, m_dis, pair_idx):











            
    dis2circs = defaultdict(set)                         
    circ2diseases = defaultdict(set)                     
    for c_idx, d_idx in pair_idx:
        dis2circs[d_idx].add(c_idx)
        circ2diseases[c_idx].add(d_idx)

                                 
                                            
    incidence_mc = []
    for h_idx, (d_idx, circs) in enumerate(dis2circs.items()):
        for c_idx in circs:
            incidence_mc.append((c_idx, h_idx))

    edge_mc = build_hyperedge_index_from_bipartite(n_circ, len(dis2circs), incidence_mc)

                                                     
    num_hye_mc = len(dis2circs)
    max_deg_mc = max((len(v) for v in dis2circs.values()), default=1)
    hye_len_mc = np.zeros(num_hye_mc, dtype=np.float32)
    hye_node_mc = np.zeros((num_hye_mc, max_deg_mc), dtype=np.int64)

    for h_idx, (d_idx, circs) in enumerate(dis2circs.items()):
        circs_sorted = sorted(list(circs))
        hye_len_mc[h_idx] = len(circs_sorted)
        for j, c_idx in enumerate(circs_sorted[:max_deg_mc]):
            hye_node_mc[h_idx, j] = c_idx + 1                    

    G_c = [
        torch.from_numpy(edge_mc).long(),
        torch.from_numpy(hye_len_mc).float(),
        torch.from_numpy(hye_node_mc).long()
    ]

                                 
                                               
    incidence_cm = []
    for h_idx, (c_idx, dset) in enumerate(circ2diseases.items()):
        for d_idx in dset:
            incidence_cm.append((d_idx, h_idx))

    edge_cm = build_hyperedge_index_from_bipartite(m_dis, len(circ2diseases), incidence_cm)

                                                     
    num_hye_cm = len(circ2diseases)
    max_deg_cm = max((len(v) for v in circ2diseases.values()), default=1)
    hye_len_cm = np.zeros(num_hye_cm, dtype=np.float32)
    hye_node_cm = np.zeros((num_hye_cm, max_deg_cm), dtype=np.int64)

    for h_idx, (c_idx, dset) in enumerate(circ2diseases.items()):
        ds_sorted = sorted(list(dset))
        hye_len_cm[h_idx] = len(ds_sorted)
        for j, d_idx in enumerate(ds_sorted[:max_deg_cm]):
            hye_node_cm[h_idx, j] = d_idx + 1                    

    G_d = [
        torch.from_numpy(edge_cm).long(),
        torch.from_numpy(hye_len_cm).float(),
        torch.from_numpy(hye_node_cm).long()
    ]

    return G_c, G_d



                                        
                                                  
                                        

def build_hy_cc(S_cc, top_n=5):
    edge_cc, hye_len_cc, hye_node_cc = build_topk_hypergraph_from_similarity(
        S_cc, top_n=top_n
    )
    G_cc = [
        torch.from_numpy(edge_cc).long(),
        torch.from_numpy(hye_len_cc).float(),
        torch.from_numpy(hye_node_cc).long()
    ]
    return G_cc


def build_disease_similarity(m_dis, pair_idx):




    dis2circs = defaultdict(set)
    for c_idx, d_idx in pair_idx:
        dis2circs[d_idx].add(c_idx)

    sim = np.zeros((m_dis, m_dis), dtype=np.float32)
    for i in range(m_dis):
        for j in range(i, m_dis):
            s1 = dis2circs[i]
            s2 = dis2circs[j]
            if not s1 and not s2:
                S_ij = 0.0
            else:
                inter = len(s1 & s2)
                union = len(s1 | s2)
                S_ij = inter / union if union > 0 else 0.0
            sim[i, j] = sim[j, i] = S_ij
    return sim


def build_hy_mm(S_dd, top_n=5):
    edge_mm, hye_len_mm, hye_node_mm = build_topk_hypergraph_from_similarity(
        S_dd, top_n=top_n
    )
    G_dd = [
        torch.from_numpy(edge_mm).long(),
        torch.from_numpy(hye_len_mm).float(),
        torch.from_numpy(hye_node_mm).long()
    ]
    return G_dd


                                        
                                    
                                        

def main():
                                                                                
                                
    DOID_OBO_FILE = r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\doid.obo.txt"
    CIRC_SIM_FILE = r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\circRNA_seq_similarity_circRNA2vec.txt"
    DISEASE_IDS_FILE = r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\disease_ids_unique_313.txt"
    ENTRIES_XLSX_FILE = r"D:\circ-DiseaseDataBase\CircR2Disease-v2.0-main\CircR2Disease-v2.0-main\The circRNA-disease entries.xlsx"

    CMI_FILES = [
        r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\CMI\CMI-9859.csv",
        r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\CMI\CMI-9905.csv",
        r"C:\Users\YuZN\PycharmProjects\DHGCDA\data\CMI\CMI-20208.csv",
    ]

    circRNAs, circ_to_idx = load_circ_ids("data/circBaseSequence_1502.csv")
    n_circ = len(circRNAs)



                                           







    S_cc = np.loadtxt(CIRC_SIM_FILE, delimiter="\t")

    assert S_cc.shape == (n_circ, n_circ), f"circ_sim 形状 {S_cc.shape} 与 Nc={n_circ} 不一致！"

                                                                           
                                                      
    diseases, disease_to_idx = load_disease_ids(DISEASE_IDS_FILE)

    m_dis = len(diseases)

                                                
    df = load_circ_disease_entries(ENTRIES_XLSX_FILE)
    df = filter_by_circ_ids(df, circ_to_idx, disease_to_idx)

                         
    dis_in_data = sorted(df["disease"].unique().tolist())
    print(f"[INFO] 关联表中实际出现的疾病种类数: {len(dis_in_data)} / {m_dis}")

                         
    pair_idx = build_pairs(df, circ_to_idx, disease_to_idx)

                     
    G_c, G_d = build_hy_mc_cm(n_circ, m_dis, pair_idx)

                                               
    G_cc = build_hy_cc(S_cc, top_n=5)

    S_dd = build_disease_similarity(m_dis, pair_idx)
    G_dd = build_hy_mm(S_dd, top_n=5)

                              
    S_cc = S_cc.astype(np.float32)                     
    S_dd = S_dd.astype(np.float32)                     
                                    
    cmi_files = CMI_FILES         
    mir2circs = load_circ_mirna_edges(cmi_files, circ_to_idx)
    G_mc, miRNA_ids = build_hy_mi(n_circ, mir2circs, max_deg_mi=50, min_deg_mi=2)

                                                                              
    md_csv_file = os.path.join("data", "miRNA-disease association data.csv")
    if not os.path.exists(md_csv_file):
        raise FileNotFoundError(f"Missing miRNA–disease CSV file: {md_csv_file}")

                                                                        
                                                    
    G_md = build_hy_do_from_mirna_disease_csv(
        disease_ids_file=DISEASE_IDS_FILE,
        md_csv_file=md_csv_file,
        max_deg_do=80,
        min_deg_do=2,
    )

                                                                                                         
    doid_ids = []


                                                                              
    md_csv_file = os.path.join("data", "miRNA-disease association data.csv")
    disease_ids_file = os.path.join("data", "disease_ids_unique_313.txt")
    if not os.path.exists(md_csv_file):
        raise FileNotFoundError(f"Missing miRNA–disease csv: {md_csv_file}")

    G_md = build_hy_do_from_mirna_disease_csv(
        disease_ids_file=disease_ids_file,
        md_csv_file=md_csv_file,
        max_deg_do=80,
        min_deg_do=2,
    )
    hyg_dict = {
        "hy_cm": G_d,
        "hy_mc": G_c,
        "hy_cc_top5": G_cc,
        "hy_mm_top5": G_dd,
        "lg_c2": S_cc,
        "lg_m2": S_dd,
        "circ_ids": circRNAs,
        "disease_ids": diseases,
        "hy_mi": G_mc,
        "miRNA_ids": miRNA_ids,
        "hy_do": G_md,

    }

    os.makedirs("hyg_new", exist_ok=True)
    out_path0 = os.path.join("hyg_new", "hyg_dict0.pickle")
    out_path1 = os.path.join("hyg_new", "hyg_dict1.pickle")
    out_path2 = os.path.join("hyg_new", "hyg_dict2.pickle")
    out_path3 = os.path.join("hyg_new", "hyg_dict3.pickle")
    out_path4 = os.path.join("hyg_new", "hyg_dict4.pickle")
    with open(out_path0, "wb") as f:
        pickle.dump(hyg_dict, f)
    with open(out_path1, "wb") as f:
        pickle.dump(hyg_dict, f)
    with open(out_path2, "wb") as f:
        pickle.dump(hyg_dict, f)
    with open(out_path3, "wb") as f:
        pickle.dump(hyg_dict, f)
    with open(out_path4, "wb") as f:
        pickle.dump(hyg_dict, f)


    print(f"[DONE] 新超图已保存到: {out_path0}")
    print(f"[INFO] Nc={n_circ}, Nd={m_dis}, 关联数={len(pair_idx)}")


if __name__ == "__main__":
    main()
