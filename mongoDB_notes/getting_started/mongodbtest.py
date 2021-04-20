# Getting started with Python and MongoDB tutorial 
# https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb

from pymongo import MongoClient
from pprint import pprint 

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://pgg1610:pushkar1610@cluster0.wlvbo.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db = client.admin

# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)