from flask import Flask, request, jsonify

app = Flask(__name__)


def read_db(password: str, name: str):
    with open('db.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 去掉换行符
            if not line:  # 跳过空行
                continue
            bd_password, bd_name = line.split(',')
            if password == bd_password and name == bd_name:
                return True
        return False


@app.route('/')
def index():
    # url传的参数
    return 'Hello World!'


# http://127.0.0.1:8080/home?name=zhangsan&gender=nan
@app.route('/getmd', methods=['POST'])
def getmd():
    if request.json is None or request.json == {}:
        return "status:101,传入数据格式错误"
    password = request.json['password']
    name = request.json['name']
    if not read_db(password, name):
        return "status:102,账号密码错误"
    return "status:200"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
