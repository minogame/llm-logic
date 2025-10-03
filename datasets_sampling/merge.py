import os
import pickle


def merge_sampled_pickles(num_files: int = 100,
                         input_template: str = "datasets_sampling/bool_logic_dataset_train_345_grpo_sampling_{idx}_sampled.pkl",
                         output_path: str = "datasets_sampling/bool_logic_dataset_train_345_grpo_sampling_merged.pkl") -> None:
    """Merge sampled pickle files where each file contains a dict mapping keys -> list.

    For each key across all files, extend a single list in the merged dict.

    Args:
        num_files: number of indexed files to try to read (0..num_files-1).
        input_template: format string containing {idx} for the index.
        output_path: path to write the merged pickle.
    """
    merged_data: dict = {}

    for idx in range(num_files):
        if idx % 10 == 0:
            print(f"Processing file {idx}")

        path = input_template.format(idx=idx)
        if not os.path.exists(path):
            print(f"File not found, skipping: {path}")
            continue

        with open(path, "rb") as f:
            data_part = pickle.load(f)

        if not isinstance(data_part, dict):
            raise TypeError(f"Expected a dict in {path}, got {type(data_part)}")

        for k, v in data_part.items():
            if v is None:
                # allow missing/None as empty
                continue
            if not isinstance(v, list):
                raise TypeError(f"Expected list for key {k} in {path}, got {type(v)}")
            merged_data.setdefault(k, []).extend(v)

    # save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"Merged dataset saved to {output_path}")


if __name__ == "__main__":
    merge_sampled_pickles()