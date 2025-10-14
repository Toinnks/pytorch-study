from openai import OpenAI
import base64
from flask import Flask, request, jsonify
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 设置 OpenAI 的 API 密钥和 API 基础地址
openai_api_key = "EMPTY"
openai_api_base = ("http://172.16.252.144:11400/v1")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

# 定义提示模型
prompt_model ='''在当前施工环境的图片中，请按照以下步骤进行分析：
**精确识别机械设备**：
- 识别图片中的挖掘机 
**检测机械下方人员**：
- 检查每台机械设备的下方、正前方、正后方区域是否存在人员站立或活动
**判断危险行为**：
- 如果发现任何人员处于机械设备的危险作业区域,如起重臂下方,判定为"true"
**最终输出结果**：
- 严格以JSON格式输出最终结果，要求只包含以下字段:```json{"has_person_under_machine": "false"#"true"}```'''


# 读取图片并转换为 Base64 编码
def image_to_base64(image_path):
    """将图片文件转换为Base64编码"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode("utf-8")
        return base64_string
    except Exception as e:
        logger.error(f"图片转换失败: {e}")
        return None


def analyze_image_by_path(image_path):
    """根据图片路径调用模型分析"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        logger.error(f"文件不存在: {image_path}")
        return {"error": f"文件不存在: {image_path}"}, 404

    # 检查是否为文件
    if not os.path.isfile(image_path):
        logger.error(f"不是有效文件: {image_path}")
        return {"error": f"不是有效文件: {image_path}"}, 400

    # 转换图片为Base64
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return {"error": "图片转换失败"}, 500

    base64_image = f"data:image;base64,{base64_image}"

    try:
        # 调用模型分析
        chat_response = client.chat.completions.create(
            model="qwen-7b",
            messages=[
                {"role": "system", "content": "你是一名经验丰富的工地安全监控专家，擅长通过图像识别危险行为和违规操作。"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_model
                        }
                    ]
                }
            ]
        )

        # 提取模型返回结果
        result = chat_response.choices[0].message.content
        logger.info(f"模型分析结果: {result}")

        # 尝试解析JSON结果（如果模型返回的是JSON格式）
        try:
            import json
            parsed_result = json.loads(result)
            return parsed_result, 200
        except:
            # 如果解析失败，直接返回原始结果
            return {"result": result}, 200

    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        return {"error": f"模型调用失败: {e}"}, 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """分析指定路径的图片"""
    try:
        # 从请求JSON中获取图片路径
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({"error": "缺少必要参数: image_path"}), 400

        image_path = data['image_path']
        logger.info(f"收到分析请求: {image_path}")

        # 调用分析函数
        result, status_code = analyze_image_by_path(image_path)
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"请求处理失败: {e}")
        return jsonify({"error": f"请求处理失败: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=15671, debug=True)