from analyses import CohesionAnalyses
from utils.utility_functions import get_locations_list


def main():
    publication_locations = ['ACL', 'EMNLP', 'ARXIV']
    discourse_connectives_filepath = 'data/connectives/discourse_connectives.txt'
    discourse_connectives_base_path = 'data/connectives/' # 'data/connectives/' for all or 'data/connectives/[venue]/' (e.g. 'data/connectives/acl/')
    locations = get_locations_list()
    analyses = CohesionAnalyses(locations, discourse_connectives_base_path)
    analyses.get_all_papers("data/nlp_abstract_data.csv", publication_locations)
    analyses.calculate_connective_counts(discourse_connectives_filepath)
    analyses.calculate_cohesive_devices_per_sentence()


if __name__ == '__main__':
    main()
