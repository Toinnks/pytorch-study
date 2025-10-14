import json
import time

import redis
import uuid
from flask import Flask, request,jsonify

app =Flask(__name__)

#http://127.0.0.1:8080/home?name=zhangsan&gender=nan
@app.route('/createTask',methods=['POST'])
def create_task():
    rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    task_id = str(uuid.uuid4())
    task_data=request.json['data']
    task_ls=[task_id,task_data]
    rs.lpush("task_list",json.dumps(task_ls))
    return jsonify({"status": 200, "task_id": task_id, "message": "任务已创建"})
@app.route('/getResult',methods=['POST'])
def get_result():
    rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    task_id=request.form['task_id']
    result_data=rs.hget('result_list',task_id)
    if not result_data:
        return "任务还未完成"
    rs.hdel("result_list",task_id)
    return json.dumps({"status":200,"data":result_data})
def get_task():
    rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    task=rs.brpop("task_list",timeout=10)
    if task is None:
        return
    return task
def complete_task():
    while True:
        task=get_task()
        if task is None:
            print('没有任务继续等待')
            continue
        task_id,task_data=task
        result_data=f"success：{task_data}"
        time.sleep(5)
        rs = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        rs.hset('result_hash', task_id, result_data)


if __name__ == '__main__':
    complete_task()
    app.run(host='127.0.0.1',port=8080)