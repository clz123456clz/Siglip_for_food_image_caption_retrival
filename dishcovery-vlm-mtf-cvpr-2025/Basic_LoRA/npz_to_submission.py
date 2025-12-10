import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pandas as pd

num_captions = 4650
num_images = 3253

def npz_to_csv(filename):
    sparse_matrix = load_npz(filename)
    row_indices, col_indices = sparse_matrix.nonzero()

    coordinates = np.vstack((row_indices, col_indices)).T
    
    save_path = filename.replace('.npz', '.csv')
    np.savetxt(save_path, coordinates, delimiter=',', fmt='%d', header='row,col', comments='')
    
    
    print(f'✅ csv saved to {save_path}')

def csv_to_npz(filename):
    coordinates = np.loadtxt(filename, delimiter=',', skiprows=1)
    row_indices, col_indices = coordinates.T
    
    data = np.ones(len(row_indices), dtype=int)
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_images, num_captions))

    save_path = filename.replace('.csv','.npz')
    save_npz(save_path, sparse_matrix)
    
    print(f'✅ npz saved to {save_path}')

    
def npz_to_submission(filename):
    gt_matrix = load_npz(filename)
    coo = gt_matrix.tocoo()
    row_indices, col_indices = coo.row, coo.col

    data = dict.fromkeys(np.arange(num_images), '')

    for i, row_idx in enumerate(row_indices):
        col_idx = col_indices[i]
        if data[row_idx] == '':
            data[row_idx] = str(col_idx)
        else:
            data[row_idx] += f"-{col_idx}"

    # print(data)
    df = pd.DataFrame({'image_id': data.keys(), 'class_ids': data.values()})
    df.to_csv('./results/sigliplarge384_single_lora_proj_head_en_v2_t2_r16_qv.csv', index=False)

    save_path = filename.replace('.npz', '.csv')
    
    print(f'✅ csv saved in kaggle submission format: {save_path}')

filename1 = "./results/sigliplarge384_multi_lora_proj_head_en_v2_t2_r16_qv.npz"
filename2 = "./results/sigliplarge384_single_lora_proj_head_en_v2_t2_r16_qv.npz"

npz_to_submission(filename2)