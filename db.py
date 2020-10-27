import sqlite3
from sqlite3 import Error
import exp2,audit

def make_db(in_path,db_path):
	results=exp2.show_result(in_path,acc=True)
	conn = sqlite3.connect(db_path)
	sql_str='''CREATE TABLE results (id integer,
	                                common text,
		                            deep text,
		                            clf text,
		                            hard integer,
		                            acc  real)'''
	try:
		conn.execute(sql_str)
	except:
		print("table exist")
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

def show( db_path,table="results"):
	conn = sqlite3.connect(db_path)
	sql_str="SELECT * FROM %s" % table
	for row in conn.execute(sql_str):
		print(row)

def make_acc(in_path,db_path):
	acc=audit.show_acc(in_path)
	conn = sqlite3.connect(db_path)
	n_cats=len(acc[0])
	names=",".join(['cat%d integer'%i 
			for i in range(n_cats)])	
	sql_str='''CREATE TABLE acc (id real,%s)''' % names
	try:
		conn.execute(sql_str)
	except:
  		print("table exist")
	for i,acc_i in enumerate(acc):
		acc_i=",".join([ str(cat_j) 
					for cat_j in acc_i])
		sql_str_i="INSERT INTO acc VALUES (%d,%s)" % (i,acc_i)
		conn.execute(sql_str_i)
	conn.commit()
	conn.close()

if __name__ == '__main__':
#	make_db("votes/MSR","db/result.db")
	show("db/result.db","acc")
#	make_acc("votes/MSR","db/result.db")