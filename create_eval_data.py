from data import read_classification_data

# for all the classification categories
categories = []
for i in range(0, len(categories), 3):
    a_name = categories[i]
    b_name = categories[i + 1]
    c_name = categories[i + 2]
    a, b, c = read_classification_data(a_name, b_name, c_name, subset_size=1000)
    a.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")
    b.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")
    c.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")

# for all the QA categories
categories = ['SearchQA', 'TriviaQA', 'HotpotQA', 'SQuAD', 'NaturalQuestions', 'BioASQ', 'RelationExtraction', 'TextbookQA', 'DuoRC']
for i in range(0, len(categories), 3):
    a_name = categories[i]
    b_name = categories[i + 1]
    c_name = categories[i + 2]
    a, b, c = read_classification_data(a_name, b_name, c_name, subset_size=1000)
    a.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")
    b.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")
    c.save_to_disk(f"eval_data/classification/{a_name, b_name, c_name}")