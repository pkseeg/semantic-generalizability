from data import read_classification_data, read_example_data
from model import read_olmo, read_qwen3b, read_qwen05b
from embed import embed
from measure import depth


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
    dist = depth(a_, b_)

    # 2. set up M_A as M specialized in A (either via ICL, RAG, SFT, or DPO)

    # 3. evaluate M_A on B and C


    # If (across lots of iterations of A, B, and C) 
    # the scores of M_A and B and C correlate with distances between A, B and A, C
    # We can know whether LLMs are able to semantically generalize



if __name__ == "__main__":
    main()

