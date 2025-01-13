from data import read_classification_data
from model import olmo_model, olmo_tokenizer
from embed import embed


def main():
    # Experiment steps
    a, b, c = read_classification_data("raw_review_Kindle_Store", "raw_review_Books", "raw_review_Amazon_Fashion")

    # 1. measure distance between A, B and A, C using M embedding strategy
    a_ = embed(a, olmo_model, olmo_tokenizer)

    # 2. set up M_A as M specialized in A (either via ICL, RAG, SFT, or DPO)

    # 3. evaluate M_A on B and C


    # If (across lots of iterations of A, B, and C) 
    # the scores of M_A and B and C correlate with distances between A, B and A, C
    # We can know whether LLMs are able to semantically generalize

