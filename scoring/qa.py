def exact_match(ytrues, yhats):
    num_match = 0
    for i in range(len(yhats)):
        pred = yhats[i]
        for ytrue in ytrues[i]:
            if ytrue == pred:
                num_match += 1
                break
    return num_match / len(yhats)
