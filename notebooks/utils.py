from collections import Counter
from collections import defaultdict
import io
import numpy as np
import os
import pandas as pd
import pickle
import time
from tqdm import tqdm
from torchtext.vocab import Vocab
import torch


def build_vocab_from_seqs(
    prod_seqs, tokenizer, extra=["<pad>", "<bos>", "<eos>", "<unk>"], index=-1
):
    counter = Counter()
    for seq in prod_seqs:
        src, tgt = " ".join(seq[0]), " ".join(seq[1])
        if index == -1:
            counter.update(tokenizer(src))
            counter.update(tokenizer(tgt))
        else:
            counter.update(tokenizer(seq[index]))
    return Vocab(counter, specials=extra)


def build_vocab_from_file(filepath, tokenizer, index=-1):
    """
    Use this function when the file contains only text separated by a tab
    where the first element is the input and the second element is the target.
    """
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            if index == -1:
                counter.update(tokenizer(string_.strip().split("\t")[0]))
                counter.update(tokenizer(string_.strip().split("\t")[1]))
            else:
                counter.update(tokenizer(string_.strip().split("\t")[index]))
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process_meta(all_seqs, tokenizer, vocab, test_flag=False):
    """
    all_seqs is a dict with keys: 'prod', 0, 1, ..., num_prod_dim-1
    """
    data = []
    num_examples = len(all_seqs["prod"])
    num_prod_dim = len(all_seqs) - 1
    for ii in range(num_examples):
        if test_flag:
            items = all_seqs["prod"][ii]
            meta = [
                all_seqs[kk][ii] for kk in range(num_prod_dim)
            ]  # rest of the attributes for input

            src = items
            src_tensor = torch.tensor([vocab[token] for token in src], dtype=torch.long)
            meta_tensor = torch.tensor(meta)
            if src_tensor.dim() == 1:
                src_tensor = torch.unsqueeze(src_tensor, 0)
            # print(src_tensor, meta_tensor)
            src_tensor = torch.cat([src_tensor, meta_tensor])
            data.append(src_tensor)
        else:
            items = all_seqs["prod"][ii]
            meta = [
                all_seqs[kk][ii][0] for kk in range(num_prod_dim)
            ]  # rest of the attributes for input
            src, tgt = items[0], items[1]
            src_tensor = torch.tensor([vocab[token] for token in src], dtype=torch.long)
            tgt_tensor = torch.tensor([vocab[token] for token in tgt], dtype=torch.long)
            meta_tensor = torch.tensor(meta)
            if src_tensor.dim() == 1:
                src_tensor = torch.unsqueeze(src_tensor, 0)
            src_tensor = torch.cat([src_tensor, meta_tensor])
            data.append((src_tensor, tgt_tensor))
    return data


def data_process_no_meta(all_seqs, vocab, test_flag=False):
    data = []
    num_examples = len(all_seqs["prod"])
    for ii in range(num_examples):
        if test_flag:
            src = all_seqs["prod"][ii]
            src_tensor = torch.tensor([vocab[token] for token in src], dtype=torch.long)
            data.append(src_tensor)
        else:
            src, tgt = all_seqs["prod"][ii]  # already tokenized
            src_tensor = torch.tensor([vocab[token] for token in src], dtype=torch.long)
            tgt_tensor = torch.tensor([vocab[token] for token in tgt], dtype=torch.long)
            data.append((src_tensor, tgt_tensor))
    return data


def data_process_from_file(filepath, tokenizer, vocab, test_flag=False):
    raw_iter = iter(io.open(filepath, encoding="utf8"))
    data = []
    for raw in raw_iter:
        if test_flag:
            src = raw.strip()
            src_tensor = torch.tensor(
                [vocab[token] for token in tokenizer(src)], dtype=torch.long
            )
            data.append(src_tensor)
        else:
            src, tgt = raw.strip().split("\t")
            src_tensor = torch.tensor(
                [vocab[token] for token in tokenizer(src)], dtype=torch.long
            )
            tgt_tensor = torch.tensor(
                [vocab[token] for token in tokenizer(tgt)], dtype=torch.long
            )
            data.append((src_tensor, tgt_tensor))
    return data


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


"""
{'product_code': 663713,
'prod_name': 'Atlanta Push Body Harlow',
'product_type_no': 283,
'product_type_name': 'Underwear body',
'product_group_name': 'Underwear',
'graphical_appearance_no': 1010016,
'graphical_appearance_name': 'Solid',
'colour_group_code': 9,
'colour_group_name': 'Black',
'perceived_colour_value_id': 4,
'perceived_colour_value_name': 'Dark',
'perceived_colour_master_id': 5,
'perceived_colour_master_name': 'Black',
'department_no': 1338,
'department_name': 'Expressive Lingerie',
'index_code': 'B',
'index_name': 'Lingeries/Tights',
'index_group_no': 1,
'index_group_name': 'Ladieswear',
'section_no': 61,
'section_name': 'Womens Lingerie',
'garment_group_no': 1017,
'garment_group_name': 'Under-, Nightwear',
'detail_desc': 'Lace push-up body with underwired, moulded, padded cups for a larger bust and fuller cleavage. Narrow, adjustable shoulder straps, an opening with a hook-and-eye fastening at the back and a lined gusset with concealed press-studs.'}

"""


def write_train_file(
    data_dir, file_name, customer_list, transactions, item_dict, prod_dict, **kwargs
):
    pattern = "%Y-%m-%d %H:%M:%S"
    colsep = kwargs.get("colsep", "\t")
    write_session_info = kwargs.get("write_session_info", False)
    original_product_name = kwargs.get("original_product_name", False)
    write_product_meta = kwargs.get("write_product_meta", True)

    count = 0
    user_id = 0
    seq_lens = []
    out_file = os.path.join(data_dir, file_name)

    include_meta = [
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    ]

    with open(out_file, "w") as fw:
        for cust in tqdm(customer_list):
            user_id += 1

            if write_session_info:
                all_items = transactions[cust]["products"]
                num_items = [len(x) for x in all_items]
                session_ids = flatten_list(
                    [[ii] * jj for ii, jj in enumerate(num_items)]
                )
                all_dates = transactions[cust]["days"]
                items = flatten_list(all_items)
                dates = flatten_list(all_dates)
            else:
                items = transactions[cust]["products"][0]
                dates = transactions[cust]["days"][0]

            if original_product_name:
                prods = [ii for ii in items]
            else:
                prods = [item_dict[ii] for ii in items]

            epochs = [
                int(time.mktime(time.strptime(str(date_time), pattern)))
                for date_time in dates
            ]  # in seconds
            epochs = [int((e - epochs[0]) / 86400.0) for e in epochs]

            if write_session_info:
                epochs = session_ids

            seq_lens.append(len(prods))

            if write_product_meta:
                meta_info = {}
                meta_info["user_id"] = user_id
                meta_info["article_id"] = prods
                for jj in include_meta:
                    data_ij = [prod_dict[ii][jj] for ii in items]
                    meta_info[jj] = data_ij

                meta_info["epochs"] = epochs
                meta_info = pd.DataFrame(meta_info)
                meta_info.to_csv(fw, sep=colsep, header=False, index=False)
                count += len(meta_info)

            else:
                # only product-id and time
                for p, t in zip(prods, epochs):
                    fw.write(colsep.join([str(user_id), str(p), str(t)]) + "\n")
                    count += 1

    print(
        f"Written {count} lines in {out_file}, {user_id} users and {len(item_dict)} items"
    )
    print(
        f"Sequence length, min: {np.min(seq_lens)}, averag: {np.mean(seq_lens):.2f} and max: {np.max(seq_lens)}"
    )


def get_session_data(inp_file, **kwargs):
    """
    inp_file contains tab-separated user-product interaction data sorted with
    respect to time. Time is represented as sessions enumerated starting with
    0 and all products in the same session get the same session number. There
    could be other columns representing different product related information.

    """
    colsep = kwargs.get("colsep", "\t")
    inp_seq_len = kwargs.get("inp_seq_len", 12)
    tgt_seq_len = kwargs.get("tgt_seq_len", 12)
    convert_to_integer = kwargs.get("convert_to_integer", True)

    def get_ids(elems):
        ids = []
        for ii, e in enumerate(elems):
            if e not in prod_dict[ii]:
                prod_dict[ii][e] = len(prod_dict[ii]) + 1
            ids.append(prod_dict[ii][e])
        return ids

    def break_sessions(seqs):
        sids = sorted(list(set([x[-1] for x in seqs])))
        temp = [[] for _ in range(len(sids))]
        for seq in seqs:
            temp[seq[-1]].append(seq[:-1])
        return temp

    sample = pd.read_csv(inp_file, sep=colsep, nrows=5)
    ncol = sample.shape[1]

    num_prod_dim = ncol - 3  # other than u, i, t
    if num_prod_dim > 0:
        prod_dict = [{} for _ in range(num_prod_dim)]
    else:
        prod_dict = None

    User = defaultdict(list)
    with open(inp_file, "r") as fr:
        for line in tqdm(fr):
            if ncol == 3:
                u, i, _ = line.rstrip().split(colsep)
            elif ncol >= 4:
                elems = line.rstrip().split(colsep)
                u, i, t = elems[0], elems[1], elems[-1]
                pdims = elems[2:-1]
                pids = get_ids(pdims)
            u = int(u)
            if convert_to_integer:
                i = int(i)
            t = int(t)
            if ncol >= 4:
                User[u].append([i] + pids + [t])
            else:
                User[u].append(i)
    print(f"Read {len(User)} user interactions")

    if num_prod_dim > 0:
        all_seqs = {k: [] for k in range(num_prod_dim)}
        all_seqs["prod"] = []  # one more dictionary for the products

    for u in User:
        seqs = break_sessions(User[u])
        for ii in range(1, len(seqs)):
            inp, tgt = seqs[ii - 1], seqs[ii]
            if len(inp) > inp_seq_len:
                inp = inp[-inp_seq_len:]  # taking the last 12
            if len(tgt) > tgt_seq_len:
                tgt = tgt[:tgt_seq_len]  # taking the first 12
            inp_p = [str(ii[0]) for ii in inp]  # only the product-id
            tgt_p = [str(ii[0]) for ii in tgt]  # always only the product-id
            all_seqs["prod"].append((inp_p, tgt_p))
            for jj in range(num_prod_dim):
                inp_jj = [kk[jj + 1] for kk in inp]
                tgt_jj = [kk[jj + 1] for kk in tgt]
                all_seqs[jj].append((inp_jj, tgt_jj))

    return all_seqs, prod_dict


def get_session_data_test(inp_file, prod_dict, **kwargs):
    """
    inp_file contains tab-separated user-product interaction data sorted with
    respect to time. Time is represented as sessions enumerated starting with
    0 and all products in the same session get the same session number. There
    could be other columns representing different product related information.

    The input file is same as in get_session_data(), prod_dict is generated
    by the same function and reused here.

    The test data contains only the last session (and maybe one more) and no
    target. This is for evaluation and scoring.

    """
    colsep = kwargs.get("colsep", "\t")
    inp_seq_len = kwargs.get("inp_seq_len", 12)

    def get_ids(elems):
        ids = []
        for ii, e in enumerate(elems):
            ids.append(prod_dict[ii][e])
        return ids

    def break_sessions(seqs):
        sids = sorted(list(set([x[-1] for x in seqs])))
        temp = [[] for _ in range(len(sids))]
        for seq in seqs:
            temp[seq[-1]].append(seq[:-1])
        return temp

    sample = pd.read_csv(inp_file, sep=colsep, nrows=5)
    ncol = sample.shape[1]

    num_prod_dim = ncol - 3  # other than u, i, t
    User = defaultdict(list)
    with open(inp_file, "r") as fr:
        for line in tqdm(fr):
            if ncol == 3:
                u, i, _ = line.rstrip().split(colsep)
            elif ncol >= 4:
                elems = line.rstrip().split(colsep)
                u, i, t = elems[0], elems[1], elems[-1]
                pdims = elems[2:-1]
                pids = get_ids(pdims)
            u = int(u)
            i = int(i)
            t = int(t)
            if ncol >= 4:
                User[u].append([i] + pids + [t])
            else:
                User[u].append(i)
    print(f"Read {len(User)} user interactions")

    if num_prod_dim > 0:
        all_seqs = {k: [] for k in range(num_prod_dim)}
        all_seqs["prod"] = []  # one more dictionary for the products

    for u in User:
        seqs = break_sessions(User[u])
        inp = seqs[-1]
        if len(inp) > inp_seq_len:
            inp = inp[-inp_seq_len:]  # taking the last 12
        inp_p = [str(ii[0]) for ii in inp]  # only the product-id
        all_seqs["prod"].append((inp_p))
        for jj in range(num_prod_dim):
            inp_jj = [kk[jj + 1] for kk in inp]
            all_seqs[jj].append((inp_jj))

    return all_seqs


def create_submission_file(
    res_dir,
    res_path,
    submit_file,
    item_dict,
    original_customers,
    extra_custs,
    dummy_pred,
):
    def id2string(row):
        return " ".join(["0" + str(ii) for ii in row])

    with open(os.path.join(res_dir, res_path), "rb") as fr:
        pred = pickle.load(fr)

    print(pred.shape)
    inv_item_dict = {v: k for k, v in item_dict.items()}

    pred2 = np.copy(pred)
    for k in tqdm(inv_item_dict):
        pred2[pred == k] = inv_item_dict[k]
    pred3 = np.apply_along_axis(id2string, 1, pred2)
    pred3 = list(pred3)

    pred_extra = [dummy_pred] * len(extra_custs)

    pred_all = pred3 + pred_extra
    all_custs = original_customers + list(extra_custs)
    res_df = pd.DataFrame({"customer_id": all_custs, "prediction": pred_all})

    res_df.to_csv(os.path.join(res_dir, submit_file), header=True, index=False)

    return res_df


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
