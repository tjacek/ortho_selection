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
		print(r_i)
		r_i=r_i.split(",")
		print(r_i)
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

def make_acc(in_path,db_path,
				acc_type="cat",name="acc"):
	acc=audit.show_acc(in_path,acc_type=acc_type)
	db=DB(db_path)
	n_cats=len(acc[0])
	data=",".join(['cat%d integer'%i 
			for i in range(n_cats)])	
	data="id integer,%s" % data
	db.make(name,data)
	for i,acc_i in enumerate(acc):
		acc_i=",".join([ str(cat_j) 
					for cat_j in acc_i])
		acc_i="%d,%s" % (i,acc_i)
		db.insert(name,acc_i)
	db.close()

def make_indv_cat(in_path,db_path,n_cats=20):
	acc=audit.show_acc(in_path,acc_type="indv_cat")
	db=DB(db_path)
	data=",".join(['cat%d integer'%i 
			for i in range(n_cats)])
	data="id integer,cat integer,%s" % data
	db.make("indv_cat",data)
	for i,acc_i in enumerate(acc):
		for j,acc_j in enumerate(acc_i):
			acc_j=",".join([ str(cat_j) 
					for cat_j in acc_j])
			acc_j="%d,%d,%s" % (i,j,acc_j)
			db.insert("indv_cat",acc_j)
	db.close()

def cat_query(cat,db_path,name="acc",threshold=0.85):
	conn = sqlite3.connect(db_path)
	cols=" %s.cat,%s.cat%d,results.*" % (name,name,cat)
	cond=" %s.cat%d>%f" % (name,cat,threshold)
	query_template(cols,name,cond,conn)

def query_template(cols,name,cond,conn):
	sql_str='''SELECT %s
				FROM %s
				INNER JOIN results
				ON %s.id=results.id
				WHERE %s'''
	sql_str=sql_str % (cols,name,name,cond)
	print(sql_str) 
	for row in conn.execute(sql_str):
		print(row)

def query(db_path,max_thres=0.81):
	conn = sqlite3.connect(db_path)
	sql_str='''SELECT results.*,acc_indv.*
				FROM acc_indv
				INNER JOIN results
				ON acc_indv.id==results.id '''
	for row in conn.execute(sql_str):
		if(max(row[7:])<max_thres):
			print(row)

if __name__ == '__main__':
#	make_db("votes/MSR2","db/result.db")
#	show("db/result.db","results")
#	make_acc("votes/MSR","db/result.db",
#		name="acc_indv",acc_type="indv")
	cat_query(6,"db/result.db",name="indv_cat")
#	make_indv_cat("votes/MSR2","db/result.db")