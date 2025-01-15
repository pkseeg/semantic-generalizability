from data import read_classification_data, read_example_data
from model import read_olmo, read_qwen3b, read_qwen05b
from embed import embed
from measure import depth, info_gain
from specialize.ICL import ICLModel
from scoring.classification import f1


def main(a_name, b_name, c_name, dev = False):
    # Experiment steps
    if dev:
        a, b, c = read_example_data(a_name, b_name, c_name) #"example_data/kindle_subset", "example_data/books_subset", "example_data/fashion_subset"
        model, tokenizer = read_qwen05b()
    else:
        a, b, c = read_classification_data(a_name, b_name, c_name) #"raw_review_Kindle_Store", "raw_review_Books", "raw_review_Amazon_Fashion"
        model, tokenizer = read_olmo()

    # 1. measure distance between A, B and A, C using M embedding strategy
    a_ = embed(a, model, tokenizer)
    b_ = embed(b, model, tokenizer)
    c_ = embed(c, model, tokenizer)
    dist_b = info_gain(a_, b_)
    dist_c = info_gain(a_, c_)

    # 2. set up M_A as M specialized in A (either via ICL, RAG, SFT, or DPO)
    icl = ICLModel(model, tokenizer)
    icl.specialize(a)

    # 3. evaluate M_A on B and C
    ytrues_b, yhats_b = icl.predict_classification(b)
    ytrues_c, yhats_c = icl.predict_classification(c)
    f1_b = f1(ytrues_b, yhats_b)
    f1_c = f1(ytrues_c, yhats_c)

    print(f"Distance between A, B: {dist_b}\nScore of ICL A on B: {f1_b}")
    print(f"Distance between A, C: {dist_c}\nScore of ICL A on C: {f1_c}")



if __name__ == "__main__":
    main()

