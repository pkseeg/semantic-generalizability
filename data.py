from datasets import load_dataset

def process_classification(ds):
    def process_example(example):
        return {
            "text": example["title"] + "\n" + example["text"],
            "label": example["rating"]
        }
    
    return ds.map(process_example, remove_columns=ds.column_names)

def read_classification_data(a_name, b_name, c_name):
    dataset_list = []
    for name in [a_name, b_name, c_name]:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name, split="full[:5%]")
        processed_ds = process_classification(dataset)
        dataset_list.append(processed_ds)
    return dataset_list



if __name__ == "__main__":
    a, b, c = read_classification_data("raw_review_Kindle_Store", "raw_review_Books", "raw_review_Amazon_Fashion")
    print(a[0])
    print(b[0])
    print(c[0])