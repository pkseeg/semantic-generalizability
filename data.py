from datasets import load_dataset, load_from_disk

def process_classification(ds):
    def process_example(example):
        return {
            "text": example["title"] + "\n" + example["text"],
            "label": example["rating"]
        }
    
    return ds.map(process_example, remove_columns=ds.column_names)

def read_classification_data(a_name, subset_size = 500):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", a_name, split="full", download_mode="force_redownload")
    dataset = dataset.shuffle(seed=42).select(range(min(subset_size, len(dataset))))
    processed_ds = process_classification(dataset)
    return processed_ds

def read_example_data(a_name, b_name, c_name):
    a = load_from_disk(a_name)
    b = load_from_disk(b_name)
    c = load_from_disk(c_name)
    return a, b, c



if __name__ == "__main__":
    a = read_classification_data("raw_review_Arts_Crafts_and_Sewing")
    print(a[0])
    
    # Save datasets to disk
    a.save_to_disk("example_data/arts_crafts_sewing_subset")
    