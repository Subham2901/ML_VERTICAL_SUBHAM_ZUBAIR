import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DIR, PROCESSED_DIR, WAVEFORM_START_COL_INDEX, RANDOM_SEED, TEST_SIZE

def load_folder(folder:Path,label:int) -> tuple[list[np.ndarray], list[int], list[str]]:
    X_list:list[np.ndarray]=[]
    y_list:list[int]=[]
    file_list:list[str]=[]
    csv_files=sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in the {folder }")
        return X_list,y_list,file_list
    # As the data is purely numeric with no headers -> header =None
    for fp in csv_files:
        df=pd.read_csv(fp, header=None)

        if df.shape[1]<=WAVEFORM_START_COL_INDEX:
            raise ValueError(
                f"{fp.name}: Expected at least {WAVEFORM_START_COL_INDEX+1} columns, got{df.shape[1]}"
            )
        # Now we will take the waveform from column R onward (row-wise)
        X=df.iloc[:,WAVEFORM_START_COL_INDEX:].to_numpy(dtype=np.float32)

        # Now as each row is one measurement
        X_list.append(X)
        y_list.extend([label]*X.shape[0])
        file_list.append(f"{fp.name} (rows={X.shape[0]},cols]{X.shape[1]})")

        print(f"Loaded {fp.name}:(rows={X.shape[0]})")

    return X_list,y_list,file_list

def main():
    person_dir=RAW_DIR/"person"
    object_dir=RAW_DIR/"object"

    PROCESSED_DIR.mkdir(parents= True,exist_ok=True)

    # Load both the classes
    Xp_list,yp_list,person_files = load_folder(person_dir,label=1)
    Xo_list, yo_list, object_files= load_folder(object_dir,label=0)

    if not Xp_list or not Xo_list:
        raise RuntimeError(
            "Missing data, Please ensure you have CSVs in both data/raw/person and data/raw/object"
        )
    #Concatenation of all rows from all files
    X_person = np.vstack(Xp_list)
    X_object = np.vstack(Xo_list)

    X=np.vstack([X_person,X_object]).astype(np.float32)
    y=np.array(yp_list+yo_list,dtype=np.int64)

    #Sanity Checks
    if X.shape[0]!=y.shape[0]:
        raise RuntimeError(f"Shape mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}")
    
    print("\n===DATASET SUMMARY===")
    print("X shape: ",X.shape)
    print("y shape: ",y.shape)
    print("Class counts: ",{0:int((y==0).sum()), 1: int((y==1).sum())})
    print("Feature Column (waveform length): ", X.shape[1])

    #Train-test split (Stratified)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_SEED,stratify=y)
    # Save to disk

    np.save(PROCESSED_DIR/"X_train.npy",X_train)
    np.save(PROCESSED_DIR/"X_test.npy",X_test)
    np.save(PROCESSED_DIR/"y_train.npy",y_train)
    np.save(PROCESSED_DIR/"y_test.npy",y_test)


    meta = {
        "waveform_start_col_idx": WAVEFORM_START_COL_INDEX,
        "X_shape_full": [int(X.shape[0]), int(X.shape[1])],
        "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "X_test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "class_counts_full": {
            "0_object": int((y == 0).sum()),
            "1_person": int((y == 1).sum())
        },
        "source_files_person": person_files,
        "source_files_object": object_files,
        "dtype": "float32",
        "note": "Features extracted from column R onward (0-based idx 17)."
    }

    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n Saved processed dataset to:", PROCESSED_DIR)
    print("Files: X_train.npy, X_test.npy, y_train.npy, y_test.npy, meta.json")

if __name__ == "__main__":
    main()


