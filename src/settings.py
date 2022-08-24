import os
from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
EXP_DIR = ROOT_DIR / "experiments"
NB_DIR = ROOT_DIR / "notebooks"


HUMANIZE = {
    "score": "NDCG",
    "ndcg": "NDCG",
    "ndcg@10": "NDCG@10",
    "ndcg@50": "NDCG@50",
    "ndcg@100": "NDCG@100",
    "ndcg@150": "NDCG@150",
    "ranking_baseline": "Baseline",
    "ranking_antigenicity": "Ranking",
}


PROT2SPECIES = {
    "UP000001432": "Actinobacillus pleuropneumoniae",
    "UP000002676": "Bordetella pertussis",
    "UP000002565": "Brucella abortus",
    "UP000000799": "Campylobacter jejuni",
    "UP000000800": "Chlamydia muridarum",
    "UP000002230": "Edwardsiella tarda",
    "UP000000625": "Escherichia coli",
    "UP000006743": "Haemophilus parasuis",
    "UP000007841": "Klebsiella pneumoniae",
    "UP000001584": "Mycobacterium tuberculosis",
    "UP000000425": "Neisseria meningitidis",
    "UP000002438": "Pseudomonas aeruginosa",
    "UP000006931": "Rickettsia prowazekii",
    "UP000008962": "Salmonella typhimurium",
    "UP000001884": "Shigella flexneri",
    "UP000006386": "Staphylococcus aureus",
    "UP000000586": "Streptococcus pneumoniae",
    "UP000000750": "Streptococcus pyogenes",
    "UP000000811": "Treponema pallidum",
    "UP000326807": "Yersinia pestis",
    "benchmark": "Benchmark"
}

OUTER = ["outer membrane", "extracellular space"]


def infer_species(proteome):
    return PROT2SPECIES[proteome]