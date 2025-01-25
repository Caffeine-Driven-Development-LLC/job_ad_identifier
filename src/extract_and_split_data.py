import numpy as np
import nltk

import dao.career_listings_db_client as db_client
import util.file_utils as file_utils

# get the first argument from the command line
import sys

if len(sys.argv) != 2:
    raise ValueError("The path to the database must be provided as an argument")

db_path = sys.argv[1]
db_client.open_database(db_path)

# Should the train data have an equal number of positive and negative examples?
train_classifications_sum_even = False

train_data_percentage = 0.8
validation_data_percentage = 0.1
test_data_percentage = 0.1

if train_data_percentage + validation_data_percentage + test_data_percentage != 1:
    raise ValueError("The sum of the data split percentages must be equal to 1")

# get the data from the database
total_hyperlinks = db_client.get_hyperlink_row_count()
hyperlink_data = db_client.get_hyperlink_data()

# split the data into job postings and non job postings
job_postings = [d for d in hyperlink_data if d[4] == 1]
non_job_postings = [d for d in hyperlink_data if d[4] == 0]

# remove the 4th element from each data point
job_postings = [[d[0], d[1], d[2], d[3], d[5], d[6]] for d in job_postings]
non_job_postings = [[d[0], d[1], d[2], d[3], d[5], d[6]] for d in non_job_postings]


# remove any new line characters from the data
def sanitize_rows(data):
    return [sanitize_row(d)
            for d in data]


def remove_repeating_whitespace(str):
    import re
    return re.sub(' +', ' ', str)


def remove_special_characters(str):
    import re
    return re.sub('[^A-Za-z0-9]+', ' ', str)


def undo_pascal_case(str):
    import re
    return re.sub(r'([A-Z])', r' \1', str)


def sanitize_string(str):
    return remove_special_characters(undo_pascal_case(remove_repeating_whitespace(str))).lower().strip()


def sanitize_row(d):
    from urllib.parse import urlparse
    url_row = urlparse(d[2]).path
    inner_text_row = d[3]
    role_title_row = d[4] if d[4] else ""
    location_row = d[5] if d[5] else ""
    return [sanitize_string(url_row), sanitize_string(inner_text_row),
            sanitize_string(role_title_row), sanitize_string(location_row)]


job_postings = sanitize_rows(job_postings)
non_job_postings = sanitize_rows(non_job_postings)

# create the word_index
nltk.download("popular")
from nltk.corpus import words

# get the english words
english_words = words.words()

word_index = set()
word_index.add("[UKN]")
word_index.add("[NUM]")

for d in job_postings + non_job_postings:
    for w in d[0].split(" "):
        if (w in english_words):
            if (w not in word_index):
                word_index.add(w)
    for w in d[1].split(" "):
        if (w in english_words):
            if (w not in word_index):
                word_index.add(w)

# save the word index to a file
word_index = list(word_index)
word_index.sort()
file_utils.save_word_index_to_file(word_index)

def get_index(word):
    # if word is a number, return the index of [NUM]
    if word.isdigit():
        return word_index.index("[NUM]")
    # if word is not in the word index, return the index of [UKN]
    if word not in word_index:
        return word_index.index("[UKN]")
    # return the index of the word
    return word_index.index(word)


def map_to_word_index(data):
    # create an array of 0s with the length of the word index
    word_index_array = np.zeros(len(word_index))
    # loop through each word in the data
    for word in data.split(" "):
        # get the index of the word
        index = get_index(word)
        # set the index of the word to 1
        word_index_array[index] = 1
    return list(map(int, word_index_array))


job_postings = [[map_to_word_index(d[0]), map_to_word_index(d[1])] for d in job_postings]
non_job_postings = [[map_to_word_index(d[0]), map_to_word_index(d[1])] for d in non_job_postings]

# shuffle the data
np.random.shuffle(job_postings)
np.random.shuffle(non_job_postings)

# get the total number of job postings and non job postings
total_job_postings = len(job_postings)
total_non_job_postings = len(non_job_postings)
percent_job_postings = round((total_job_postings / total_hyperlinks) * 100, 2)

# print relevant information
print(f"Total hyperlinks: {total_hyperlinks}")
print(f"Total job postings: {total_job_postings}")
print(f"Total non job postings: {total_non_job_postings}")
print(f"Percent job postings: %{percent_job_postings}")

# get totals for each data split
total_train_job_postings = int(total_job_postings * train_data_percentage)
total_train_non_job_postings = total_train_job_postings if train_classifications_sum_even else \
    int(total_non_job_postings * train_data_percentage)

total_validation_job_postings = int(total_job_postings * validation_data_percentage)
total_validation_non_job_postings = total_validation_job_postings if train_classifications_sum_even else \
    int(total_non_job_postings * validation_data_percentage)

total_test_job_postings = int(total_job_postings * test_data_percentage)
total_test_non_job_postings = total_non_job_postings - (
        total_train_non_job_postings + total_validation_non_job_postings) if train_classifications_sum_even else \
    int(total_non_job_postings * test_data_percentage)

# split data into train, validation, and test data
train_job_postings = job_postings[:total_train_job_postings]
train_non_job_postings = non_job_postings[:total_train_non_job_postings]

validation_job_postings = job_postings[
                          total_train_job_postings:total_train_job_postings + total_validation_job_postings]
validation_non_job_postings = non_job_postings[
                              total_train_non_job_postings:total_train_non_job_postings + total_validation_non_job_postings]

test_job_postings = job_postings[total_train_job_postings + total_validation_job_postings:]
test_non_job_postings = non_job_postings[total_train_non_job_postings + total_validation_non_job_postings:]

# save the data to files

file_utils.save_data_to_files("train", train_job_postings, train_non_job_postings)
file_utils.save_data_to_files("validation", validation_job_postings, validation_non_job_postings)
file_utils.save_data_to_files("test", test_job_postings, test_non_job_postings)
