# createsqlite3database.py

import argparse
import sqlite3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_file", help="sqlite3 database file")

    args = parser.parse_args()

    database_connection = sqlite3.connect(args.db_file)
    db_cursor = database_connection.cursor()
    db_cursor.execute('CREATE TABLE comparison(binary, function, comparebinary, comparefunction, similarityscore)')
    database_connection.commit()
    database_connection.close()

if __name__ == '__main__':
    main()
