import sqlite3

con = None


def open_database(path):
    global con
    con = sqlite3.connect(path)


def get_hyperlink_row_count():
    cursor = con.cursor()
    cursor.execute("select count(*) from hyperlinks")
    return cursor.fetchone()[0]


def get_hyperlink_data():
    cursor = con.cursor()
    cursor.execute("select id, careerPageId, url, innerText, isJobPosting, jobTitle, jobLocation from hyperlinks")
    return cursor.fetchall()
