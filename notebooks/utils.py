import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from numpy import copy


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
    # original_product_name = kwargs.get("original_product_name", False)
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
