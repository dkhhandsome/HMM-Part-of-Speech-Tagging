import os
import sys
import argparse
import numpy
import numpy as np
from collections import Counter

test_tag = [
    "N",
    "M",
    "V",
    "PUN",
]

pos_tags_collection = [
    "AJ0",
    "AJC",
    "AJS",
    "AT0",
    "AV0",
    "AVP",
    "AVQ",
    "CJC",
    "CJS",
    "CJT",
    "CRD",
    "DPS",
    "DT0",
    "DTQ",
    "EX0",
    "ITJ",
    "NN0",
    "NN1",
    "NN2",
    "NP0",
    "ORD",
    "PNI",
    "PNP",
    "PNQ",
    "PNX",
    "POS",
    "PRF",
    "PRP",
    "PUL",
    "PUN",
    "PUQ",
    "PUR",
    "TO0",
    "UNC",
    "VBB",
    "VBD",
    "VBG",
    "VBI",
    "VBN",
    "VBZ",
    "VDB",
    "VDD",
    "VDG",
    "VDI",
    "VDN",
    "VDZ",
    "VHB",
    "VHD",
    "VHG",
    "VHI",
    "VHN",
    "VHZ",
    "VM0",
    "VVB",
    "VVD",
    "VVG",
    "VVI",
    "VVN",
    "VVZ",
    "XX0",
    "ZZ0",
    "AJ0-AV0",
    "AJ0-VVN",
    "AJ0-VVD",
    "AJ0-NN1",
    "AJ0-VVG",
    "AVP-PRP",
    "AVQ-CJS",
    "CJS-PRP",
    "CJT-DT0",
    "CRD-PNI",
    "NN1-NP0",
    "NN1-VVB",
    "NN1-VVG",
    "NN2-VVZ",
    "VVD-VVN",
    "AV0-AJ0",
    "VVN-AJ0",
    "VVD-AJ0",
    "NN1-AJ0",
    "VVG-AJ0",
    "PRP-AVP",
    "CJS-AVQ",
    "PRP-CJS",
    "DT0-CJT",
    "PNI-CRD",
    "NP0-NN1",
    "VVB-NN1",
    "VVG-NN1",
    "VVZ-NN2",
    "VVN-VVD",
]


def initialize(filename):
    words = {}
    pos_tags = {}
    sentence_starts = {}
    transition_counts = {}
    total_counts = {}

    previous_pos_tag = None
    total_sentences = 0

    with open(filename, "r") as file:
        new_sentence = True
        for line in file:
            word, pos_tag = line.strip().split(" : ")

            # counting observation probabilities
            # -----------------------
            if word not in words:
                words[word] = pos_tag
            if pos_tag not in pos_tags:
                pos_tags[pos_tag] = [word]
            else:
                pos_tags[pos_tag].append(word)
            # ------------------------

            # counting initial_probabilities
            # --------------------
            if new_sentence:
                sentence_starts[pos_tag] = sentence_starts.get(pos_tag, 0) + 1
                total_sentences += 1
                new_sentence = False

            if word == "." and pos_tag == "PUN":
                new_sentence = True
            # -------------------

            # counting transition probability
            if previous_pos_tag is not None:
                if previous_pos_tag not in transition_counts:
                    transition_counts[previous_pos_tag] = {pos_tag: 1}
                    total_counts[previous_pos_tag] = 1
                else:
                    transition_counts[previous_pos_tag][pos_tag] = \
                        transition_counts[previous_pos_tag].get(pos_tag, 0) + 1
                    total_counts[previous_pos_tag] += 1

            previous_pos_tag = pos_tag

    # after we have done, add remaining char into individual table

    transition_probabilities = {
        pos_tag: {next_pos_tag: count / total_counts[pos_tag]
                  for next_pos_tag, count in next_pos_tags.items()}
        for pos_tag, next_pos_tags in transition_counts.items()}

    initial_probabilities = {pos_tag: count / total_sentences for pos_tag, count
                             in sentence_starts.items()}

    observation_probabilities = {
        pos_tag: {word: count / len(word_counts) for word, count in
                  Counter(word_counts).items()}
        for pos_tag, word_counts in pos_tags.items()
    }

    return initial_probabilities, transition_probabilities, observation_probabilities


def read_file_and_split_sentences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    sentences = []
    sentence = []

    for line in lines:
        if line == '.':
            sentences.append(' '.join(sentence))
            sentence = []
        else:
            sentence.append(line)

    return sentences


def viterbi_algorithm(sentence, initial_probabilities, transition_probabilities,
                      observation_probabilities, tags):
    # split the sentence
    words_collection = sentence.split()

    # Initialize the viterbi and backpointer matrices, where viterbi stands for
    # prob, backpointer stands for prob in pseudocode
    viterbi = {}
    backpointer = {}
    for pos_tag in tags:
        viterbi[pos_tag] = [0] * len(words_collection)
        backpointer[pos_tag] = [None] * len(words_collection)

    # smoother factor to deal with case that never appeared in training file
    e = 1e-10

    # First step: Calculate the first column of the viterbi matrix
    for pos_tag in initial_probabilities:
        emission_prob = observation_probabilities.get(pos_tag, {}).get(words_collection[0], e)
        viterbi[pos_tag][0] = initial_probabilities[pos_tag] * emission_prob

    # Calculate the rest of the viterbi matrix
    for word_idx in range(1, len(words_collection)):
        for pos_tag in tags:
            max_prob = -1
            max_pos_tag = None

            emission_prob = observation_probabilities.get(pos_tag, {}).get(words_collection[word_idx], e)

            for prev_pos_tag in tags:
                if prev_pos_tag in transition_probabilities and pos_tag in transition_probabilities[prev_pos_tag]:
                    prob = viterbi[prev_pos_tag][word_idx - 1] * transition_probabilities[prev_pos_tag][pos_tag] * emission_prob
                else:
                    prob = 0

                if prob > max_prob:
                    max_prob = prob
                    max_pos_tag = prev_pos_tag

            viterbi[pos_tag][word_idx] = max_prob
            backpointer[pos_tag][word_idx] = max_pos_tag

    # Find the most likely last POS tag
    max_prob = -1
    max_pos_tag = None
    for pos_tag in tags:
        if viterbi[pos_tag][-1] > max_prob:
            max_prob = viterbi[pos_tag][-1]
            max_pos_tag = pos_tag

    # Back trace to find the most likely sequence of POS tags
    prediction_sequence = [(words_collection[-1], max_pos_tag)]
    for word_idx in range(len(words_collection) - 1, 0, -1):
        max_pos_tag = backpointer[max_pos_tag][word_idx]
        prediction_sequence.insert(0, (words_collection[word_idx - 1], max_pos_tag))

    return prediction_sequence


def concatenate_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        first_line = True
        for file_name in input_files:
            with open(file_name, 'r') as infile:
                for line in infile:
                    stripped_line = line.strip()
                    if stripped_line:  # Check if the line is not empty
                        if not first_line:
                            outfile.write("\n")  # Add a newline character before each non-empty line
                        outfile.write(stripped_line)
                        first_line = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = [item for sublist in args.trainingfiles for item in sublist]
    # print(training_list)
    combined_training_file = "combined_training.txt"
    concatenate_files(training_list, combined_training_file)
    test_file = args.testfile
    output_file = args.outputfile

    initial_probabilities, transition_probabilities, observation_probabilities = initialize(
        combined_training_file)
    sentence_list = read_file_and_split_sentences(test_file)
    # print("error not here")
    # count = 0
    # for i in observation_probabilities.keys():
    #     count += 1
    #     print(i)
    # print(count)
    # print(len(pos_tags_collection))
    # print(compare_files("training2.txt", "finald.txt"))
    ans = []
    for i in range(len(sentence_list)):
        ans_lice = viterbi_algorithm(sentence_list[i], initial_probabilities,
                                     transition_probabilities,
                                     observation_probabilities,
                                     pos_tags_collection)
        ans.extend(ans_lice)
        ans.extend([(".", "PUN")])
    with open(output_file, 'w') as file:
        for word, pos_tag in ans:
            file.write(f"{word} : {pos_tag}\n")



    # traininglist = ["training1.txt", "training2.txt", "training1.txt", "training1.txt", "training1.txt"]
    # concatenate_files(traininglist, "fun.txt")


