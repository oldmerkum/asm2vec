# save_data.py

import contextlib
import sqlite3

def save_results_to_db(results):
    with contextlib.closing(sqlite3.connect("database.sqlite3")) as connection:
        with contextlib.closing(connection.cursor()) as cursor:
            # {binary, functionname, binarycompare, comparefunctionname, similarityscore}
            cursor.executemany("INSERT INTO comparison VALUES(?, ?, ?, ?, ?)", results)
            connection.commit()

