from analyses import MorphologicalAnalyses
from utils.utility_functions import get_locations_list


def main():
    publication_locations = ['ACL', 'EMNLP', 'ARXIV']
    locations = get_locations_list()
    analyses = MorphologicalAnalyses(locations)
    analyses.get_all_papers("data/nlp_abstract_data.csv", publication_locations)
    analyses.calculate_root_verb_tense_diversity(plot_verb_tense_dist=True)


if __name__ == '__main__':
    main()
