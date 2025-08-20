import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    # FIXED: Use adaptive padding based on data size
    total_length = data.shape[1]
    if total_length <= 500:  # Short time series like Walmart data
        padding = min(10, total_length // 10)  # Use 10% of data or max 10
    else:
        padding = 200  # Original padding for long time series
    
    print(f"Using padding={padding} for time series of length {total_length}")
    
    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    for pred_len in pred_lens:
        print(f"Evaluating prediction horizon: {pred_len}")
        
        # Check if we have enough data for this prediction horizon
        train_steps_after_pred = train_data.shape[1] - pred_len
        valid_steps_after_pred = valid_data.shape[1] - pred_len
        test_steps_after_pred = test_data.shape[1] - pred_len
        
        if train_steps_after_pred <= padding or valid_steps_after_pred <= 0 or test_steps_after_pred <= 0:
            print(f"  Skipping pred_len={pred_len} - insufficient data")
            continue
            
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        if train_features.shape[0] == 0 or valid_features.shape[0] == 0 or test_features.shape[0] == 0:
            print(f"  Skipping pred_len={pred_len} - no samples generated")
            continue
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        # FIXED: Simplified inverse transform handling
        original_pred_shape = test_pred.shape
        original_labels_shape = test_labels.shape
        
        # Reshape to 2D for scaler
        test_pred_2d = test_pred.reshape(-1, test_pred.shape[-1])
        test_labels_2d = test_labels.reshape(-1, test_labels.shape[-1])
        
        # Apply inverse transform
        test_pred_inv_2d = scaler.inverse_transform(test_pred_2d)
        test_labels_inv_2d = scaler.inverse_transform(test_labels_2d)
        
        # Reshape back to original 4D shape
        test_pred_inv = test_pred_inv_2d.reshape(original_pred_shape)
        test_labels_inv = test_labels_inv_2d.reshape(original_labels_shape)
            
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res