import sqlite3
from sqlite3 import Error
import exp2,audit

class DB(object):
	def __init__(self,db_path):
		self.conn=sqlite3.connect(db_path)

	def make(self,name,data):
		sql_str='''CREATE TABLE %s (%s)'''
		sql_str=sql_str % (name,data)
		try:
			self.conn.execute(sql_str)
		except:
			print("table exist")

	def insert(self,name,data_i):
		sql_str="INSERT INTO %s VALUES (%s)" % (name,data_i)
		self.conn.execute(sql_str)

	def close(self):
		self.conn.commit()
		self.conn.close()

def make_db(in_path,db_path):
	results=exp2.show_result(in_path,acc=True)
	db=DB(db_path)
	sql_str='''id integer,
				common text,
				deep text,
				clf text,
				hard integer,
				acc  real'''
	db.make("results",sql_str)
	for i,r_i in enumerate(results):
		r_i=r_i.split(",")
		hard_i=int(r_i[3]=='True')
		tuple_i= (i,r_i[0],r_i[1],r_i[2],hard_i,float(r_i[4]))
		data_i="%d,'%s','%s','%s',%d,%f" % tuple_i
		db.insert("results",data_i)
	db.close()

def show( db_path,table="results"):
	conn = sqlite3.connect(db_path)
	sql_str="SELECT * FROM %s" % table
	for row in conn.execute(sql_str):
		print(row)

def make_acc(in_path,db_path):
	acc=audit.show_acc(in_path)
	db=DB(db_path)
	n_cats=len(acc[0])
	data=",".join(['cat%d integer'%i 
			for i in range(n_cats)])	
	data="id integer,%s" % data
	db.make("acc",data)
	for i,acc_i in enumerate(acc):
		acc_i=",".join([ str(cat_j) 
					for cat_j in acc_i])
		acc_i="%d,%s" % (i,acc_i)
		db.insert("acc",acc_i)
	db.close()

def query(cat,db_path,threshold=0.5):
	conn = sqlite3.connect(db_path)
	sql_str='''SELECT acc.cat%d,results.*
				FROM acc
				INNER JOIN results
				ON acc.id==results.id 
				WHERE acc.cat%d>%f'''
	sql_str= sql_str % (cat,cat,threshold)
	for row in conn.execute(sql_str):
		print(row)

if __name__ == '__main__':
#	make_db("votes/MSR","db/result.db")
#	show("db/result.db","acc")
#	make_acc("votes/MSR","db/result.db")
	query(19,"db/result.db")