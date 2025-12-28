import os
import glob
import cv2
import numpy as np
import pandas as pd

def extract_window_rgb_summary(image_dir_path, save_csv_path=None):
    """
    Summarizes RGB gradation statistics for all .jpg images in a directory.
    Target Window: Y[60:74], X[0:255] (15 Rows)
    Performs row-wise normalization and aggregation.
    
    Output: DataFrame with columns:
    [FileName, RowID, Channel, Value_Abs_Avg, Value_Abs_Std, Value_Norm_Avg, Value_Norm_Std]
    """
    
    summary_dfs = []
    
    # 1. Search for image files
    image_files = sorted(glob.glob(os.path.join(image_dir_path, "*.jpg")))
    print(f"Found {len(image_files)} images in {image_dir_path}")

    for file_path in image_files:
        try:
            file_name = os.path.basename(file_path)
            
            # 2. Load Image
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Crop Window
            # Y: 60~74 (15 rows) -> Slice: 60:75
            # X: 0~255 -> Slice: 0:256
            roi = img_rgb[60:75, 0:256] # Shape: (15, 256, 3)
            
            rows, cols, _ = roi.shape # rows=15, cols=256
            
            # Row IDs (Absolute Y coordinates)
            row_ids = np.arange(60, 60 + rows)
            
            # Function for Row-wise Normalization
            def get_norm_matrix(channel_data):
                mat = channel_data.astype(float)
                mins = mat.min(axis=1, keepdims=True)
                maxs = mat.max(axis=1, keepdims=True)
                rng = maxs - mins
                rng[rng == 0] = 1.0
                return (mat - mins) / rng

            channels = ['R', 'G', 'B']
            channel_indices = [0, 1, 2]
            
            for ch, ch_idx in zip(channels, channel_indices):
                # Extract channel data (15, 256)
                data_abs = roi[:, :, ch_idx]
                data_norm = get_norm_matrix(data_abs)
                
                # Calculate Statistics (Aggregation)
                # Groupby equivalent using numpy axis ops
                abs_avg = data_abs.mean(axis=1)
                abs_std = data_abs.std(axis=1)
                norm_avg = data_norm.mean(axis=1)
                norm_std = data_norm.std(axis=1)
                
                # Create DataFrame for this channel
                df_ch = pd.DataFrame({
                    'FileName': file_name,
                    'RowID': row_ids,
                    'Channel': ch,
                    'Value_Abs_Avg': abs_avg,
                    'Value_Abs_Std': abs_std,
                    'Value_Norm_Avg': norm_avg,
                    'Value_Norm_Std': norm_std
                })
                
                summary_dfs.append(df_ch)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    # 4. Merge and Save
    if summary_dfs:
        final_df = pd.concat(summary_dfs, ignore_index=True)
        
        # Ensure column order
        cols_order = ['FileName', 'RowID', 'Channel', 
                      'Value_Abs_Avg', 'Value_Abs_Std', 
                      'Value_Norm_Avg', 'Value_Norm_Std']
        final_df = final_df[cols_order]
        
        if save_csv_path:
            final_df.to_csv(save_csv_path, index=False)
            print(f"Summary data saved to: {save_csv_path}")
            
        return final_df
    else:
        print("No data extracted.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Settings
    target_dir = "/Users/petrenko/Documents/Program_study/Antigravity_Works/ImageClassifier"
    output_path = os.path.join(target_dir, "rgb_gradation_summary.csv")
    
    # Run
    df_result = extract_window_rgb_summary(target_dir, output_path)
    
    # Check
    print(df_result.head())
    print(f"Total Summary Rows: {len(df_result)}")
