import os
import numpy as np

from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)



def con_code():
    '''code copied from jupyter notebook'''
    
    text = open(os.path.join(os.getcwd(), 'data/constitution.txt')).read()
    obs, obs_map = parse_observations(text)
    hmm8 = unsupervised_HMM(obs, 10, 100)   
    print(hmm8.A)
    print(hmm8.O)
    return

if __name__ == "__main__":
    con_code()