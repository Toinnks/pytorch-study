import json
import redis
import uuid
from flask import Flask, request,jsonify
rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def get_task():
    task=rs.brpop("test_task",timeout=5)

if __name__ == '__main__':
    get_task()