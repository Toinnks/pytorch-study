import json

dict1 = {
    "name": "张三",
    "age": 25,
    "hobbies": ["篮球", "音乐"],
    "is_student": True
}
print(type(dict1))
json_string = json.dumps(dict1)
print(type(json_string))
data=json.loads(json_string)
print(type(data))
