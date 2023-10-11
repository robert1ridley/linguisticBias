from analyses import Analyses
from utils.utility_functions import get_locations_list


def get_dists():
    venue_dict = {}

    with open("data/nlp_abstract_data.csv", 'r') as f:
        lines = f.readlines()
        for line in lines:
            text_splits = line.split('\t')
            nation = text_splits[-2].strip()
            venue = text_splits[-1].strip()
            if venue.startswith('ACL'):
                venue_code = 'ACL'
            elif venue.startswith('EMNLP'):
                venue_code = 'EMNLP'
            elif venue.startswith('ARXIV'):
                venue_code = 'ARXIV'
            else:
                raise Exception

            if venue_code in venue_dict.keys():
                if nation in venue_dict[venue_code].keys():
                    venue_dict[venue_code][nation] += 1
                else:
                    venue_dict[venue_code][nation] = 1
            else:
                venue_dict[venue_code] = {
                    nation: 1
                }

    for item in venue_dict.items():
        print("VENUE: ", item[0])
        venue_sum = sum(item[1].values())
        print("SUM:", venue_sum)
        dists = [(k[0], k[1] / venue_sum) for k in item[1].items()]
        print(dists)


def get_stats():
    publication_locations = ['ACL', 'EMNLP', 'ARXIV']
    locations = get_locations_list()
    analyses = Analyses(locations)
    analyses.get_all_papers("data/nlp_abstract_data.csv", publication_locations)
    analyses.get_corpus_stats()


if __name__ == '__main__':
    get_dists()
    get_stats()
