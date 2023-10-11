from analyses import LexicalComplexity
from utils.utility_functions import get_locations_list


def main():
    publication_locations = ['ACL', 'EMNLP', 'ARXIV']
    locations = get_locations_list()
    analyses = LexicalComplexity(locations)
    analyses.get_all_papers("data/nlp_abstract_data.csv", publication_locations)
    analyses.calculate_token_type_ratio()
    analyses.get_lexical_chains()


if __name__ == '__main__':
    main()
