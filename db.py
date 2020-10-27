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
	for i,result_i in enumerate(results):
		insert(i,result_i.split(","),conn)
	conn.commit()
	conn.close()

def insert(i,r_i,conn):
	hard_i=int(r_i[3]=='True')
	tuple_i= (i,r_i[0],r_i[1],r_i[2],hard_i,float(r_i[4]))
	data_i="(%d,'%s','%s','%s',%d,%f)" % tuple_i
	sql_str="INSERT INTO results VALUES %s" % data_i
	conn.execute(sql_str)

def show( db_path):
	conn = sqlite3.connect(db_path)
	for row in conn.execute("SELECT * FROM results"):
		print(row)

if __name__ == '__main__':
#    make_db("votes/MSR","db/result.db")
	show("db/result.db")