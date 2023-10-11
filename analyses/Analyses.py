import sys
import string
import json
import claucy
import spacy
from collections import defaultdict
import seaborn as sns
from pandas import DataFrame
from os.path import exists
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from nltk import RegexpTokenizer, pos_tag


class Analyses:
    def __init__(self, filtered_countries):
        self.filtered_countries = filtered_countries
        self.all_papers = []
        self.nlp = spacy.load("en_core_web_sm")
        claucy.add_to_pipe(self.nlp)

    def get_all_papers(self, data_path, publication_venues):
        titles = []
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split = line.split('\t')
                title = split[0]
                abstract_text = split[1]
                abstract_text = abstract_text.replace('https', '')
                title = title.replace('- ACL Anthology', '').strip()
                country = split[2].strip()
                conference_name = split[3].strip()
                in_venue = False
                for ven in publication_venues:
                    if conference_name.startswith(ven):
                        in_venue = True
                if country in self.filtered_countries and title not in titles and in_venue:
                    item = {
                        'title': title,
                        'abstract_text': abstract_text,
                        'country': country,
                        'venue': conference_name
                    }
                    self.all_papers.append(item)
                    titles.append(title)

    def get_corpus_stats(self):
        counts = {}
        for i, paper in enumerate(self.all_papers):
            self.print_progress(i, len(self.all_papers))
            abstract_text = paper['abstract_text']
            country = paper['country']
            abstract_text = abstract_text.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(abstract_text)
            if country in counts.keys():
                counts[country]['papers'] += 1
                counts[country]['words'] += len(words)
            else:
                counts[country] = {
                    'papers': 1,
                    'words': len(words)
                }
        for c, stats in counts.items():
            print(c)
            print(stats)
            print()

    @staticmethod
    def print_progress(current_number, num_papers):
        print_message = "Analysing abstract " + str(current_number) + " of " + str(num_papers)
        print(print_message)
        sys.stdout.write("\033[F")

    @staticmethod
    def calculate_statistical_significance(nations_dict):
        for nat in nations_dict.keys():
            if nat != 'United States':
                t_score, p_value = ttest_ind(nations_dict['United States'],
                                             nations_dict[nat])
                print('Comparing United States vs.', nat)
                print('T-score', t_score)
                print('p-value', p_value)
                print()

    @staticmethod
    def distribution_plotter(input_dict, plot_title):
        reformatted_dict = {'nation': [], 'values': []}
        for k, v in input_dict.items():
            nation_list = [k for _ in range(len(v))]
            reformatted_dict['nation'].extend(nation_list)
            reformatted_dict['values'].extend(v)

        df = DataFrame.from_dict(reformatted_dict)
        cat = sns.catplot(x='nation', y='values', data=df, kind='boxen')
        cat.fig.suptitle(plot_title)
        plt.show()


class LexicalComplexity(Analyses):
    def __init__(self, filtered_countries):
        super().__init__(filtered_countries)
        self.nation_verb_usage_dict = {'nation': [], 'verb_tag': [], 'verb': [], 'counts': []}
        self.verb_token_type_scores = {'nation': [], 'token_type': [], 'score': []}
        self.parsed_papers = {}

    def calculate_token_type_ratio(self, debug=False, plot_distribution=False):
        all_ratios_by_nation = {}
        if debug:
            papers = self.all_papers[:40]
        else:
            papers = self.all_papers
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            country = paper['country']
            abstract = paper['abstract_text']
            abstract = abstract.lower()
            words = word_tokenize(abstract)

            total_word_count = len(words)
            unique_words = set(words)
            num_unique_words = len(unique_words)
            type_token_ratio = num_unique_words / total_word_count

            if country not in all_ratios_by_nation.keys():
                all_ratios_by_nation[country] = [type_token_ratio]
            else:
                all_ratios_by_nation[country].append(type_token_ratio)
        average_ratios = {nation: sum(all_ratios_by_nation[nation]) / len(all_ratios_by_nation[nation])
                          for nation in all_ratios_by_nation.keys()}
        print()
        print("Average Type-Token Ratios:")
        print(average_ratios)
        print()
        self.calculate_statistical_significance(all_ratios_by_nation)
        if plot_distribution:
            self.distribution_plotter(all_ratios_by_nation, "Average Type-Token Ratio per Abstract")

    def get_lexical_chains(self, debug=False):
        position = ['NN', 'NNS', 'NNP', 'NNPS']
        tokenizer = RegexpTokenizer(r'\w+')
        if debug:
            papers = self.all_papers[:40]
        else:
            papers = self.all_papers
        nation_lengths = {}
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            country = paper['country']
            input_txt = paper['abstract_text']
            sentence = sent_tokenize(input_txt)
            tokens = [tokenizer.tokenize(w) for w in sentence]
            tagged = [pos_tag(tok) for tok in tokens]
            nouns = [word.lower() for i in range(len(tagged))
                     for word, pos in tagged[i] if pos in position]

            relation_list = defaultdict(list)
            for k in range(len(nouns)):
                relation = []
                for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
                    for l in syn.lemmas():
                        relation.append(l.name())
                        if l.antonyms():
                            relation.append(l.antonyms()[0].name())
                    for l in syn.hyponyms():
                        if l.hyponyms():
                            relation.append(l.hyponyms()[0].name().split('.')[0])
                    for l in syn.hypernyms():
                        if l.hypernyms():
                            relation.append(l.hypernyms()[0].name().split('.')[0])
                relation_list[nouns[k]].append(relation)

            lexical = []
            threshold = 0.5
            for noun in nouns:
                flag = 0
                for j in range(len(lexical)):
                    if flag == 0:
                        for key in list(lexical[j]):
                            if key == noun and flag == 0:
                                lexical[j][noun] += 1
                                flag = 1
                            elif key in relation_list[noun][0] and flag == 0:
                                syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                                syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                                if syns1[0].wup_similarity(syns2[0]) >= threshold:
                                    lexical[j][noun] = 1
                                    flag = 1
                            elif noun in relation_list[key][0] and flag == 0:
                                syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                                syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                                if syns1[0].wup_similarity(syns2[0]) >= threshold:
                                    lexical[j][noun] = 1
                                    flag = 1
                if flag == 0:
                    dic_nuevo = {}
                    dic_nuevo[noun] = 1
                    lexical.append(dic_nuevo)
                    flag = 1

            final_chain = []
            while lexical:
                result = lexical.pop()
                if len(result.keys()) == 1:
                    for value in result.values():
                        if value != 1:
                            final_chain.append(result)
                else:
                    final_chain.append(result)

            chain_lengths = []
            for chain in final_chain:
                chain_lengths.append(len(chain))

            try:
                if country in nation_lengths.keys():
                    nation_lengths[country].append(sum(chain_lengths) / len(chain_lengths))
                else:
                    nation_lengths[country] = [sum(chain_lengths) / len(chain_lengths)]
            except ZeroDivisionError:
                print("ERROR")
                print(input_txt)
                print(final_chain)

        ave_nation_lengths = {k: sum(v) / len(v) for k, v in nation_lengths.items()}
        print(ave_nation_lengths)
        self.calculate_statistical_significance(nation_lengths)


class MorphologicalAnalyses(Analyses):
    def __init__(self, filtered_countries):
        super().__init__(filtered_countries)
        self.parsed_papers = {}

    def calculate_root_verb_tense_diversity(self, plot_verb_tense_dist=False, debug=False):
        verb_tag_dict_by_nation = {}
        verb_by_tag_by_nation = {}
        if debug:
            papers = self.all_papers[:200]
        else:
            papers = self.all_papers
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            country = paper['country']
            abstract = paper['abstract_text']
            doc = self.nlp(abstract)
            if country not in self.parsed_papers.keys():
                self.parsed_papers[country] = [doc]
            else:
                self.parsed_papers[country].append(doc)
            for token in doc:
                if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                    word = token.text.lower()
                    if country not in verb_tag_dict_by_nation.keys():
                        verb_tag_dict_by_nation[country] = [token.tag_]
                    else:
                        verb_tag_dict_by_nation[country].append(token.tag_)

                    if country not in verb_by_tag_by_nation.keys():
                        verb_by_tag_by_nation[country] = {}
                        verb_by_tag_by_nation[country][token.tag_] = [word]
                    else:
                        if token.tag_ not in verb_by_tag_by_nation[country].keys():
                            verb_by_tag_by_nation[country][token.tag_] = [word]
                        else:
                            verb_by_tag_by_nation[country][token.tag_].append(word)
        unique_tags = {nation: set(verb_tag_dict_by_nation[nation]) for nation in verb_tag_dict_by_nation.keys()}
        tag_counts = {}
        for nation in verb_tag_dict_by_nation.keys():
            tag_counts[nation] = {}
            for unique_tag in unique_tags[nation]:
                tag_counts[nation][unique_tag] = verb_tag_dict_by_nation[nation].count(unique_tag)
        if plot_verb_tense_dist:
            self.verb_tense_distribution_plotter(tag_counts, "Verb Form Distributions")

    @staticmethod
    def verb_tense_distribution_plotter(tag_dict, plot_title):
        reformatted_dict = {}
        for nation in tag_dict.keys():
            reformatted_dict[nation] = {'verb_type': [], 'counts': []}
            for verb_token in tag_dict[nation]:
                if verb_token in reformatted_dict[nation]['verb_type']:
                    type_index = reformatted_dict[nation]['verb_type'].index(verb_token)
                    reformatted_dict[nation]['counts'][type_index] += tag_dict[nation][verb_token]
                else:
                    reformatted_dict[nation]['verb_type'].append(verb_token)
                    reformatted_dict[nation]['counts'].append(tag_dict[nation][verb_token])

        all_normalized_counts = {'locale': [], 'verb_type': [], 'counts': []}
        for country, values in reformatted_dict.items():
            normalized_counts = [val / sum(values['counts']) for val in values['counts']]
            values['counts'] = normalized_counts
            for i, item in enumerate(values['verb_type']):
                all_normalized_counts['locale'].append(country)
                all_normalized_counts['verb_type'].append(item)
                all_normalized_counts['counts'].append(values['counts'][i])
        df = DataFrame.from_dict(all_normalized_counts)
        print(df)

        cat = sns.factorplot(x='counts', y='verb_type', hue='locale', orient='h', data=df, kind='bar')
        cat.fig.suptitle(plot_title)
        plt.show()


class SyntacticAnalyses(Analyses):
    def __init__(self, filtered_countries):
        super().__init__(filtered_countries)
        self.parsed_papers = {}

    def calculate_modifiers_per_noun_phrase(self, debug=False):
        noun_phrase_modifier_dict = {}
        if debug:
            papers = self.all_papers[:200]
        else:
            papers = self.all_papers
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            abstract = paper['abstract_text']
            nation = paper['country']
            doc = self.nlp(abstract)
            for sentence in doc.sents:
                for np in sentence.noun_chunks:
                    head_noun = np.root
                    modifiers = list(head_noun.children)
                    modifier_types = [modifier.pos_ for modifier in modifiers]
                    filtered_modifiers = [modifiers[i] for i, modifier_type in enumerate(modifier_types) if
                                          modifier_type != 'PUNCT']
                    modifier_length = len(modifiers)
                    dash_filtered_modifiers = []
                    skip_flag = False
                    for j, mod in enumerate(filtered_modifiers):
                        if skip_flag:
                            skip_flag = False
                            continue
                        if mod.text == '-' and mod.pos_ == 'ADJ':
                            if j == len(filtered_modifiers) - 1:
                                post_word = mod.text
                            else:
                                post_word = mod.text + filtered_modifiers[j + 1].text
                            if j == 0:
                                dash_filtered_modifiers.append(post_word)
                            else:
                                dash_filtered_modifiers[-1] += post_word
                            skip_flag = True
                        else:
                            dash_filtered_modifiers.append(mod.text)

                    if nation not in noun_phrase_modifier_dict.keys():
                        noun_phrase_modifier_dict[nation] = [modifier_length]
                    else:
                        noun_phrase_modifier_dict[nation].append(modifier_length)

        average_modifier_lengths = {
            nation: sum(noun_phrase_modifier_dict[nation]) / len(noun_phrase_modifier_dict[nation])
            for nation in noun_phrase_modifier_dict
        }

        print(average_modifier_lengths)
        self.calculate_statistical_significance(noun_phrase_modifier_dict)
        self.distribution_plotter(noun_phrase_modifier_dict, 'Average Noun Phrase Modifier Length')

    def get_clause_count(self, debug=False):
        if debug:
            papers = self.all_papers[:200]
        else:
            papers = self.all_papers
        clause_count_dict = {}
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            country = paper['country']
            abstract = paper['abstract_text']
            sentences = sent_tokenize(abstract)
            for sentence in sentences:
                doc = self.nlp(sentence)
                sentence_clause_count = len(doc._.clauses)
                if country not in clause_count_dict.keys():
                    clause_count_dict[country] = [sentence_clause_count]
                else:
                    clause_count_dict[country].append(sentence_clause_count)
        average_clause_counts = {
            nation: sum(clause_count_dict[nation]) / len(clause_count_dict[nation])
            for nation in clause_count_dict.keys()
        }
        print()
        print("Average Clause Counts:")
        print(average_clause_counts)
        print()
        self.calculate_statistical_significance(clause_count_dict)
        self.distribution_plotter(clause_count_dict, "Average Number of Clauses per Sentence")

    def get_sentence_parse_tree_depth(self, debug=False):
        print('Calculating average parse depth')
        country_sentence_parse_depth_dict = {}
        if debug:
            papers = self.all_papers[:200]
        else:
            papers = self.all_papers
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            abstract = paper['abstract_text']
            processed = self.nlp(abstract)
            country = paper['country']

            abstract_depths = []
            for sent in processed.sents:
                depths = {}

                def walk_tree(node, depth):
                    depths[node.orth_] = depth
                    if node.n_lefts + node.n_rights > 0:
                        return [walk_tree(child, depth + 1) for child in node.children]

                walk_tree(sent.root, 0)
                abstract_depths.append(max(depths.values()))
                if country not in country_sentence_parse_depth_dict.keys():
                    country_sentence_parse_depth_dict[country] = abstract_depths
                else:
                    country_sentence_parse_depth_dict[country].extend(abstract_depths)
        average_parse_depths = {
            nation:
                sum(country_sentence_parse_depth_dict[nation]) / len(country_sentence_parse_depth_dict[nation])
            for nation in country_sentence_parse_depth_dict.keys()}
        print("Average parse depths:")
        print(average_parse_depths)
        print()
        self.calculate_statistical_significance(country_sentence_parse_depth_dict)
        self.distribution_plotter(country_sentence_parse_depth_dict, "Average Sentence Parse Tree Depth")

    def get_average_sentence_length(self, debug=False):
        print('Calculating average sentence length')
        country_sentence_length_dict = {}
        if debug:
            papers = self.all_papers[:200]
        else:
            papers = self.all_papers
        for i, paper in enumerate(papers):
            self.print_progress(i, len(papers))
            abstract_text = paper['abstract_text']
            sentences = sent_tokenize(abstract_text)
            words = [word_tokenize(sentence) for sentence in sentences]
            sentence_lengths = [len(sentence_words) for sentence_words in words]
            country = paper['country']
            if country in country_sentence_length_dict.keys():
                country_sentence_length_dict[country].extend(sentence_lengths)
            else:
                country_sentence_length_dict[country] = sentence_lengths
        average_sentence_lengths = {
            nation: sum(country_sentence_length_dict[nation]) / len(country_sentence_length_dict[nation])
            for nation in country_sentence_length_dict
        }
        print('Average sentence lengths:')
        print(average_sentence_lengths)
        print()
        self.calculate_statistical_significance(country_sentence_length_dict)
        self.distribution_plotter(country_sentence_length_dict, "Average Sentence Length Distribution")


class CohesionAnalyses(Analyses):
    def __init__(self, filtered_countries, connectives_path):
        super().__init__(filtered_countries)
        self.connectives_count_dict_path = connectives_path + 'connectives_count_dict.json'
        self.connectives_cat_dict_path = connectives_path + 'connectives_cat_dict.json'
        self.connectives_sub_cat_dict_path = connectives_path + 'connectives_sub_cat_dict.json'
        self.connective_terms_dict_path = connectives_path + 'connective_terms_dict.json'
        self.word_counts_path = connectives_path + 'word_counts.json'
        self.connectives_to_sentences_path = connectives_path + 'connectives_to_sentences.json'
        self.connectives_count_dict = {}
        self.connectives_cat_dict = {}
        self.connectives_sub_cat_dict = {}
        self.connective_terms_dict = {}
        self.word_counts = {}
        self.connectives_to_sentences = {}

    def load_connective_dicts(self):
        with open(self.connectives_count_dict_path) as connectives_count_dict_json_file:
            self.connectives_count_dict = json.load(connectives_count_dict_json_file)
        with open(self.connectives_cat_dict_path) as connectives_cat_dict_json_file:
            self.connectives_cat_dict = json.load(connectives_cat_dict_json_file)
        with open(self.connectives_sub_cat_dict_path) as connectives_sub_cat_dict_json_file:
            self.connectives_sub_cat_dict = json.load(connectives_sub_cat_dict_json_file)
        with open(self.connective_terms_dict_path) as connective_terms_dict_json_file:
            self.connective_terms_dict = json.load(connective_terms_dict_json_file)
        with open(self.word_counts_path) as word_counts_json_file:
            self.word_counts = json.load(word_counts_json_file)
        with open(self.connectives_to_sentences_path) as connectives_to_sentences_json_file:
            self.connectives_to_sentences = json.load(connectives_to_sentences_json_file)

    def calculate_connective_counts(self, connectives_list_filepath):
        if exists(self.connectives_count_dict_path) and exists(self.connectives_cat_dict_path) and \
                exists(self.connectives_sub_cat_dict_path) and exists(self.connective_terms_dict_path) and \
                exists(self.word_counts_path):
            print('Loading Dictionaries from file')
            self.load_connective_dicts()
        else:
            print('Calculating Dictionaries')
            connectives_dict = {}
            with open(connectives_list_filepath, 'r') as connectives_file:
                lines = connectives_file.readlines()
                for line in lines:
                    line = line.strip().lower()
                    splits = line.split('\t')
                    category = splits[0]
                    sub_category = splits[1]
                    connective = splits[2]
                    if connective not in connectives_dict.keys():
                        connectives_dict[connective] = {'category': category, 'sub_category': sub_category}

            papers = self.all_papers
            for i, paper in enumerate(papers):
                self.print_progress(i, len(papers))
                abstract = paper['abstract_text'].lower()
                country = paper['country']
                abstract_sentences = sent_tokenize(abstract)
                for j, sentence in enumerate(abstract_sentences):
                    words = word_tokenize(sentence)
                    word_count = len(words)
                    if country in self.word_counts.keys():
                        self.word_counts[country] += word_count
                    else:
                        self.word_counts[country] = word_count
                    sentence_connective_count = 0
                    sentence_cat_list = []
                    sentence_sub_cat_list = []
                    sentence_connective_list = []
                    for connective in connectives_dict.keys():
                        category = connectives_dict[connective]['category']
                        sub_category = connectives_dict[connective]['sub_category']
                        connective_words = word_tokenize(connective)
                        if connective in sentence:
                            if len(connective_words) == 1:
                                if connective in words:
                                    sentence_connective_count += 1
                                    sentence_cat_list.append(category)
                                    sentence_sub_cat_list.append(sub_category)
                                    sentence_connective_list.append(connective)
                            else:
                                sentence_connective_count += 1
                                sentence_cat_list.append(category)
                                sentence_sub_cat_list.append(sub_category)
                                sentence_connective_list.append(connective)
                            if j == 0:
                                sent_range = abstract_sentences[0:j + 2]
                            else:
                                sent_range = abstract_sentences[j - 1:j + 2]
                            if country in self.connectives_to_sentences.keys():
                                if connective in self.connectives_to_sentences[country].keys():
                                    self.connectives_to_sentences[country][connective].append(
                                        sent_range)
                                else:
                                    self.connectives_to_sentences[country][connective] = [sent_range]
                            else:
                                self.connectives_to_sentences[country] = {}
                                self.connectives_to_sentences[country][connective] = [sent_range]

                    if country in self.connectives_count_dict.keys():
                        self.connectives_count_dict[country].append(sentence_connective_count)
                        self.connectives_cat_dict[country].extend(sentence_cat_list)
                        self.connectives_sub_cat_dict[country].extend(sentence_sub_cat_list)
                        self.connective_terms_dict[country].extend(sentence_connective_list)
                    else:
                        self.connectives_count_dict[country] = [sentence_connective_count]
                        self.connectives_cat_dict[country] = sentence_cat_list
                        self.connectives_sub_cat_dict[country] = sentence_sub_cat_list
                        self.connective_terms_dict[country] = sentence_connective_list
            word_counts_json = json.dumps(self.word_counts)
            connectives_count_dict_json = json.dumps(self.connectives_count_dict)
            connectives_cat_dict_json = json.dumps(self.connectives_cat_dict)
            connectives_sub_cat_dict_json = json.dumps(self.connectives_sub_cat_dict)
            connective_terms_dict_json = json.dumps(self.connective_terms_dict)
            connectives_to_sentences_json = json.dumps(self.connectives_to_sentences)

            fp_names_to_data_dict = {
                self.connectives_count_dict_path: connectives_count_dict_json,
                self.connectives_cat_dict_path: connectives_cat_dict_json,
                self.connectives_sub_cat_dict_path: connectives_sub_cat_dict_json,
                self.connective_terms_dict_path: connective_terms_dict_json,
                self.word_counts_path: word_counts_json,
                self.connectives_to_sentences_path: connectives_to_sentences_json
            }
            for fp, data in fp_names_to_data_dict.items():
                with open(fp, 'w') as outfile:
                    outfile.write(data)

    def calculate_cohesive_devices_per_sentence(self):
        mean_sent_connectives_dict = {
            nation: sum(self.connectives_count_dict[nation]) / len(self.connectives_count_dict[nation])
            for nation in self.connectives_count_dict.keys()
        }
        print(mean_sent_connectives_dict)
        self.calculate_statistical_significance(self.connectives_count_dict)
        self.distribution_plotter(self.connectives_count_dict, 'Number of Discourse Connectives per Sentence')
