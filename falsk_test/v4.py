import json
import redis
import uuid
from flask import Flask, request, jsonify

rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def get_task():
    task = rs.brpop("task_list", timeout=5)


if __name__ == '__main__':
    task_id = str(uuid.uuid4())
    task_data = 'hello'
    task_ls = [task_id, task_data]
    rs.lpush("task_list", json.dumps(task_ls))
    _, task = rs.brpop("task_list")
    task_data = json.loads(task)
    print(task_data[0], task_data[1])

    _, all_results = rs.hgetall('result_hash')
    data = json.loads(all_results)
    print(data)
