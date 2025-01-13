from datasets import load_dataset, load_from_disk

def process_classification(ds):
    def process_example(example):
        return {
            "text": example["title"] + "\n" + example["text"],
            "label": example["rating"]
        }
    
    return ds.map(process_example, remove_columns=ds.column_names)

def read_classification_data(a_name, b_name, c_name, subset_size = 500):
    dataset_list = []
    for name in [a_name, b_name, c_name]:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name, split="full")
        dataset = dataset.shuffle(seed=42).select(range(min(subset_size, len(dataset))))
        processed_ds = process_classification(dataset)
        dataset_list.append(processed_ds)
    return dataset_list

def read_example_data(a_name, b_name, c_name):
    a = load_from_disk(a_name)
    b = load_from_disk(b_name)
    c = load_from_disk(c_name)
    return a, b, c



if __name__ == "__main__":
    a, b, c = read_classification_data("raw_review_Kindle_Store", "raw_review_Books", "raw_review_Amazon_Fashion")
    print(a[0])
    print(b[0])
    print(c[0])

    # Save datasets to disk
    a.save_to_disk("example_data/kindle_subset")
    b.save_to_disk("example_data/books_subset")
    c.save_to_disk("example_data/fashion_subset")