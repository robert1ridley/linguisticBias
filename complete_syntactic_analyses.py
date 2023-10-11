from analyses import SyntacticAnalyses
from utils.utility_functions import get_locations_list


def main():
    publication_locations = ['ACL', 'EMNLP', 'ARXIV']
    locations = get_locations_list()
    analyses = SyntacticAnalyses(locations)
    analyses.get_all_papers("data/nlp_abstract_data.csv", publication_locations)
    analyses.calculate_modifiers_per_noun_phrase()
    analyses.get_clause_count()
    analyses.get_sentence_parse_tree_depth()
    analyses.get_average_sentence_length()


if __name__ == '__main__':
    main()
