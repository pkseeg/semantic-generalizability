import random

from data import read_classification_data, read_example_data, read_qa_data, read_qa_eval
from model import read_olmo, read_qwen3b, read_qwen05b
from embed import embed, embed_sbert
from measure import depth, info_gain
from specialize.ICL import ICLModel
from specialize.SFT import SFTModel
from scoring.classification import f1
from scoring.qa import exact_match


def main(a_name, b_name, c_name, task = "classification", dev = False):
    print("Reading model and dataset")
    # Experiment steps
    if dev:
        if task == "classification":
            a, b, c = read_example_data(a_name, b_name, c_name) #"example_data/kindle_subset", "example_data/books_subset", "example_data/fashion_subset"
        elif task == "qa":
            a = read_qa_eval(a_name) # FIXME this should be a full training set.
            b = read_qa_eval(b_name)
            c = read_qa_eval(c_name)
        #model, tokenizer = read_qwen05b()
        model, tokenizer = read_olmo()
    else:
        # FIXME read the A data on its own, then read b and c from the eval data
        a, b, c = read_classification_data(a_name, b_name, c_name) #"raw_review_Kindle_Store", "raw_review_Books", "raw_review_Amazon_Fashion"
        model, tokenizer = read_olmo()

    # FIXME also need to subset to only 500
    # print(f"Embedding {len(a) + len(b) + len(c)} texts with the model")
    # # 1. measure distance between A, B and A, C using M embedding strategy
    # #a_ = embed(a, model, tokenizer)
    # #b_ = embed(b, model, tokenizer)
    # #c_ = embed(c, model, tokenizer)
    # a_ = embed_sbert(a)
    # b_ = embed_sbert(b)
    # c_ = embed_sbert(c)
    # dist_b = depth(a_, b_)
    # dist_c = depth(a_, c_)

    # print(dist_b, dist_c)

    # 2. set up M_A as M specialized in A (either via ICL, RAG, SFT, or DPO)
    print(f"Specializing the model with dataset A (SFT)")

    # SFT
    sft = SFTModel(model, tokenizer, task="qa")
    sft.specialize(a)
    assert False


    # ICL
    # icl = ICLModel(model, tokenizer, task="qa")
    # icl.specialize(a)

    # 3. evaluate M_A on B and C
    print(f"Predicting B with M_A")
    ytrues_b, yhats_b = icl.predict_qa(b)
    print(ytrues_b)
    print(yhats_b)
    print(f"Predicting C with M_A")
    ytrues_c, yhats_c = icl.predict_qa(c)

    if task == "classificaion":
        f1_b = f1(ytrues_b, yhats_b)
        f1_c = f1(ytrues_c, yhats_c)
    elif task == "qa":
        exact_match_b = exact_match(ytrues_b, yhats_b)
        exact_match_c = exact_match(ytrues_c, yhats_c)

    print(exact_match_b, exact_match_c)
    # print(f"Distance between A, B: {dist_b}\nScore of ICL A on B: {f1_b}")
    # print(f"Distance between A, C: {dist_c}\nScore of ICL A on C: {f1_c}")



if __name__ == "__main__":
    main()

