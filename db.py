import sqlite3
from sqlite3 import Error
import exp2

def make_db(in_path,db_path):
	results=exp2.show_result(in_path,acc=True)
	conn = sqlite3.connect(db_path)
	sql_str='''CREATE TABLE results (id integer,
	                                common text,
		                            deep text,
		                            clf text,
		                            hard integer,
		                            acc  real)'''
	conn.execute(sql_str)
	for row in conn.execute("SELECT * FROM results"):
		print(row)

if __name__ == '__main__':
    make_db("votes/MSR","db/result.db")