import os
import csv

data_folder = "data"
out_folder = "out"


def save_data_to_files(dir, job_postings, non_job_postings):
    diir = f"{data_folder}/{dir}"
    if (os.path.exists(diir)):
        os.system(f"rm -r {diir}")
    os.makedirs(diir)
    with open(f"{diir}/job_postings_urls.csv", "w") as f:
        writer = csv.writer(f)
        for jp in job_postings:
            writer.writerow(jp[0])

    with open(f"{diir}/job_postings_text.csv", "w") as f:
        writer = csv.writer(f)
        for jp in job_postings:
            writer.writerow(jp[1])

    with open(f"{diir}/non_job_postings_urls.csv", "w") as f:
        writer = csv.writer(f)
        for njp in non_job_postings:
            writer.writerow(njp[0])

    with open(f"{diir}/non_job_postings_text.csv", "w") as f:
        writer = csv.writer(f)
        for njp in non_job_postings:
            writer.writerow(njp[1])


def save_word_index_to_file(word_index):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    import json
    word_index_json = {word: i for i, word in enumerate(word_index)}
    with open(f"{out_folder}/word_index.json", 'w') as f:
        json.dump(word_index_json, f)

def get_data(subset):
    job_postings = []
    non_job_postings = []
    with open(f"{data_folder}/{subset}/job_postings_urls.csv", "r") as f:
        for line in f:
            obj = {
                "url": list(map(int, line.strip().split(","))),
            }
            job_postings.append(obj)
    with open(f"{data_folder}/{subset}/non_job_postings_urls.csv", "r") as f:
        for line in f:
            obj = {
                "url": list(map(int, line.strip().split(","))),
            }
            non_job_postings.append(obj)

    with open(f"{data_folder}/{subset}/job_postings_text.csv", "r") as f:
        for i, line in enumerate(f):
            job_postings[i]["text"] = list(map(int, line.strip().split(",")))

    with open(f"{data_folder}/{subset}/non_job_postings_text.csv", "r") as f:
        for i, line in enumerate(f):
            non_job_postings[i]["text"] = list(map(int, line.strip().split(",")))

    labels = [1] * len(job_postings) + [0] * len(non_job_postings)
    data = job_postings + non_job_postings

    # shuffle the data but keep the relation between in index of the data and the index of the labels
    import random
    random.seed(0)
    random.shuffle(data)
    random.seed(0)
    random.shuffle(labels)

    return data, labels
