import os
import json
import random
from datasets import load_dataset

from data import read_classification_data

# # for all the classification categories
categories = [
    "raw_review_All_Beauty",
    "raw_review_Amazon_Fashion",
    "raw_review_Appliances",
    "raw_review_Arts_Crafts_and_Sewing",
    "raw_review_Automotive",
    "raw_review_Baby_Products",
    "raw_review_Beauty_and_Personal_Care",
    "raw_review_Books",
    "raw_review_CDs_and_Vinyl",
    "raw_review_Cell_Phones_and_Accessories",
    "raw_review_Clothing_Shoes_and_Jewelry",
    "raw_review_Digital_Music",
    "raw_review_Electronics",
    "raw_review_Gift_Cards",
    "raw_review_Grocery_and_Gourmet_Food",
    "raw_review_Handmade_Products",
    "raw_review_Health_and_Household",
    "raw_review_Health_and_Personal_Care",
    "raw_review_Home_and_Kitchen",
    "raw_review_Industrial_and_Scientific",
    "raw_review_Kindle_Store",
    "raw_review_Magazine_Subscriptions",
    "raw_review_Movies_and_TV",
    "raw_review_Musical_Instruments",
    "raw_review_Office_Products",
    "raw_review_Patio_Lawn_and_Garden",
    "raw_review_Pet_Supplies",
    "raw_review_Software",
    "raw_review_Sports_and_Outdoors",
    "raw_review_Subscription_Boxes",
    "raw_review_Tools_and_Home_Improvement",
    "raw_review_Toys_and_Games",
    "raw_review_Video_Games"
]
for i in range(len(categories)):
    a_name = categories[i]
    a = read_classification_data(a_name, subset_size=1000)
    a.save_to_disk(f"eval_data/classification/{a_name}")




# QA DATA
# def clean_qid(raw):
#     qids = raw.split("_")[1]
#     if "/" in qids:
#         qids = qids.split("/")
#         return qids[0]
#     else:
#         return qids

# ds = load_dataset("mrqa-workshop/mrqa")

# directories = ["dev", "test"]

# for dir in directories:
#     dir_path = f"eval_data/qa/MultiReQA/data/{dir}/"

#     for subdir in os.listdir(dir_path):
#         subdir_path = os.path.join(dir_path, subdir)
#         candidates_path = os.path.join(subdir_path, "candidates.json")

#         with open(candidates_path, "r") as f:
#             candidates = [json.loads(line) for line in f.readlines()]

#         random.seed(42)
#         samples = random.sample(candidates, min(1000, len(candidates)))

#         clean_ids = [clean_qid(sample["candidate_id"]) for sample in samples]

#         subset = ds.filter(lambda example: example["qid"] in clean_ids)

#         output_path = f"eval_data/qa/{subdir}"
#         subset.save_to_disk(output_path)

#         print(f"Processed and saved subset to {output_path}")
