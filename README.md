# Grad-Align
Source code for the papers
- Jin-Duk et al. "Grad-Align: Gradual Network Alignment via Graph Neural Networks" AAAI-22 (Student Poster Program)
- Jin-Duk et al. "On the Power of Gradual Network Alignment Using Dual-Perception Similarities". **IEEE TPAMI**

Pytorch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) package is used for the implementation of graph neural networks (GNNs).

# Dependancy

- Pytorch > 1.8
- torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
- numpy
- pands
- sklearn
- tqdm
- networkx
- matplotlib



# Running

run ``main.py`` script file

or in your prompt, type

``python main.py --graphname 'fb-tt' --k_hop 2 --mode 'not_perturbed' ``  

- --graphname can be either one dataset of the three ('fb-tt' for Facebook vs. Twitter dataset, 'douban' for Douban online vs. offline dataset, 'econ' for Econ perturbed pair.)
- Other description for each arguments are typed in '--help' arguments in main.py argparse.
- for the implemenation of the graphname 'econ', --mode should be changed to 'perturbed' instead of 'not_perturbed'


# etc.
If you need any further information, please e-mail me : jindeok6@yonsei.ac.kr
