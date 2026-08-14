import json
import os
import sys
import base64
import copy
import logging
from pathlib import Path
base_dir = str(Path(__file__).absolute().parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)
import util
import traceback
from flask import Flask, Response, jsonify, request,render_template
from flask_cors import *
from data.chain_of_thought.dataset_process import dataset_list, name2dataset_module
from models import support_models
from methods.assist import sort_methods_by_cost_time
from methods.assist import get_methods_by_dataset_and_model_name
from methods import method_name2sub_module, method_load_errors
from diagnose.diagnose import diagnosing, get_model_output
from util.fileutil import image_dict_to_base64
from util.visualizing import get_echarts_info_with_result
from util.model_tokenizer import get_cached_model_tok
from util.cache_process import (
    delete_cache,
    generate_cache_key,
    is_cache_entry_usable,
    read_cache,
    write_cache,
)
from embedding_utils.Vectordb import Interpret_vectordb
import random
import argparse
from openai import OpenAI
from util.openai_generate_data import system_prompt, str2dic
random.seed(1)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# server port
root_port = int(os.environ.get("PORT", 10051))


DATASET_DISPLAY_NAME_MAP = {
    "中医知识问答_0_shot": "Chinese Medicine QA",
    "传统文化问答_0_shot": "Traditional Culture QA",
    "关系抽取_0_shot": "Relation Extraction",
    "文本翻译_0_shot": "Translation",
    "知识库推理_0_shot": "Knowledge Base Reasoning",
    "缺字补全_chain_of_thought": "Missing Character Completion",
    "自动标点_chain_of_thought": "Punctuation Restoration",
    "释义生成_0_shot": "Paraphrase Generation",
}
DATASET_HIDDEN_NAMES = {
    "面向大模型的常识类动态知识探测与编辑数据",
}
DATASET_INTERNAL_NAME_MAP = {
    display_name: internal_name for internal_name, display_name in DATASET_DISPLAY_NAME_MAP.items()
}


def to_internal_dataset_name(dataset_name):
    if not dataset_name:
        return dataset_name
    return DATASET_INTERNAL_NAME_MAP.get(dataset_name, dataset_name)


def to_display_dataset_name(dataset_name):
    if not dataset_name:
        return dataset_name
    return DATASET_DISPLAY_NAME_MAP.get(dataset_name, dataset_name)


def build_dataset_list_for_ui():
    ui_list = []
    for data_name in dataset_list:
        internal_name = data_name.get("name", "")
        if internal_name in {"GPT4o_data", *DATASET_HIDDEN_NAMES}:
            continue
        item = dict(data_name)
        item["name"] = to_display_dataset_name(internal_name)
        item["des"] = ""
        ui_list.append(item)
    return ui_list


def add_dataset_display_name(sample, dataset_name):
    if not isinstance(sample, dict):
        return sample
    item = dict(sample)
    item["dataset_name"] = to_display_dataset_name(dataset_name)
    return item


def add_dataset_display_name_list(samples, dataset_name):
    return [add_dataset_display_name(sample, dataset_name) for sample in samples]

def process_model_name(model_name):
    temp_model_name = "/".join(model_name.split("/")[:2])
    model_path = "/".join(model_name.split("/")[2:]) if len(model_name.split("/"))>2 else ""

    if "✨default-" in model_path:
        model_path = None
    
    return temp_model_name.strip(), model_path.strip() if model_path is not None else None


def process_model_name_v2(model_name):
    """Return (logical_model_name, optional_model_path_override)."""
    if not model_name:
        return "", None

    model_name = model_name.strip()

    if model_name in support_models:
        return model_name, None

    if "/" in model_name:
        maybe_logical = model_name.rsplit("/", 1)[0]
        suffix = model_name.split("/")[-1]
        if maybe_logical in support_models and (
            "default" in suffix.lower() or "鉁╠efault" in suffix
        ):
            return maybe_logical, None

    temp_model_name = "/".join(model_name.split("/")[:2]).strip()
    model_path = "/".join(model_name.split("/")[2:]).strip() if len(model_name.split("/")) > 2 else None
    if temp_model_name in support_models:
        if model_path and ("default" in model_path.lower() or "鉁╠efault" in model_path):
            model_path = None
        return temp_model_name, model_path

    return model_name, None

class InterpretGUIFlask:
    
    def __init__(self, args):
        
        # init flask_app
        self.app = Flask(__name__, static_url_path='')
        self.app.config['JSON_AS_ASCII'] = False
        self.vecdb = Interpret_vectordb(args.vecdb_config_path)
        self.gpt4o = OpenAI(base_url=args.openai_base_url, 
                            api_key=args.openai_api_key)

        if os.path.exists(args.model_name2path):
            with open(args.model_name2path, 'r', encoding='utf-8') as f:
                self.model2path = json.load(f)
        else:
            self.model2path = None
        
        CORS(self.app, resources={r"/*": {"origins": "*"}}, send_wildcard=True)

        @self.app.get('/')
        def home():
            return render_template('index.html')

        @self.app.get('/getip')
        def getip():
            return jsonify({"data": request.host_url.rstrip("/"), "code": 200})


        @self.app.get('/loadModelByName')
        def loadModelByName():
            try:
                model_name, model_path = process_model_name_v2(request.args.get('modelName'))
                print('-------------model_name----------------\n',model_name)
                get_cached_model_tok(model_name=model_name, model_path=model_path, model2path=self.model2path)
                return jsonify({"data": "OK", "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})

        @self.app.get('/getDatasetList')
        def getDatasetList():
            try: 
                return jsonify({"data": build_dataset_list_for_ui(), "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})
            
            
        @self.app.post('/searchTopnByInput')
        def searchTopnByInput():
            jsondata = request.get_json()
            input_text = jsondata["input_text"]
            try:
                data_list = []
                try: 
                    if args.openai_api_key:
                        gptdata = self.gpt4o.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, 
                                                                            {"role": "user", "content": input_text}], model="gpt-4o")
                        print("---The data generated by GPT-4o---")
                        print(gptdata.choices[0].message.content)
                        gpt_dic = str2dic(gptdata.choices[0].message.content)
                        gpt_dic["prompts"] = [gpt_dic["prompt"]]
                        name2dataset_module["GPT4o_data"].support_template_keys = list(gpt_dic.keys())
                        gpt_dic["dataset_name"] = "GPT4o_data"
                        data_list.append(gpt_dic)
                except:
                    print("GPT4o generated data error!")
                
                
                res = self.vecdb.search(input_text)[0]
                """
                data_list.append({
                        "dataset_name": input_text+"测试 dataset_name"+str(i),
                        "dataset_type":input_text+"测试 dataset_type"+str(i),
                        "prompt":(input_text+"测试 prompt"+str(i))*5,
                        "task_aware_target_tokens":input_text+"测试 task_aware_target_tokens"+str(i),
                        # ....
                    })
                """
                for r in res:
                    r["entity"]["data"]["dataset_name"] = r["entity"]["info"]
                    if "text" in r["entity"]:
                        r["entity"]["data"]["retrieval_text"] = r["entity"]["text"]
                    data_list.append(r["entity"]["data"])
                        
                return jsonify({"data": data_list, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})
            
            
        @self.app.get('/getModelSecList')
        def getModelSecList():
            try:
                """
                model_config = [
                    {
                        "model_type":"meta-llama/Llama-2-7b-hf",
                        "model_list":[
                            {
                                "value":"defualt-llama2-7b",
                            },
                        ],
                    },
                    {
                        "model_type":"openai-community/gpt2-xl",
                        "model_list":[
                            {
                                "value":"default-gpt2-xl"
                            },
                        ],
                    },
                    {
                        "model_type":"google-bert/bert-base-uncased",
                        "model_list":[
                            {
                                "value":"default-bert-base-uncased"
                            },
                        ],
                    },    

                ]"""

                model_config = []
                for model in support_models:
                    model_dict = {
                        "model_type":model,
                        "model_list":[
                            {
                                "value":"default",
                            },
                        ],
                    }
                    model_config.append(model_dict)
                return jsonify({"data": model_config, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})

            
        @self.app.get('/getModelList')
        def getModelList():
            try:
                model_list = support_models                
                return jsonify({"data": model_list, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})
            
        @self.app.get('/getMethodListByModelName')
        def getMethodListByModelName():
            try:
                dataset_name = to_internal_dataset_name(request.args.get('dataset_name'))
                model_name, model_path = process_model_name_v2(request.args.get('model_name'))
                print('-------------dataset_name----------------\n',dataset_name)
                print('-------------model_name----------------\n',model_name)
                
                try:
                    method_list = get_methods_by_dataset_and_model_name(dataset_name=dataset_name,model_name=model_name)
                except:
                    traceback.print_exc()
                    method_list = []
                
                """
                method_list = [
                    {
                        "value": "External",
                        "label": "External",
                        "children":[
                            {
                                "value": "method_typexxx1",
                                "label": "method_typexxx1",
                                "children":[
                                    {
                                        "value": "Attribution1",
                                        "label": "Attribution1",
                                    },
                                    {
                                        "value": "111",
                                        "label": "111",
                                    },
                                ]
                            },
                            {
                                "value": "method_typexxx2",
                                "label": "method_typexxx2",
                                "children":[
                                    {
                                        "value": "Attribution2",
                                        "label": "Attribution2",
                                    },
                                    {
                                        "value": "222",
                                        "label": "222",
                                    },
                                ]
                            }
                        ]
                    },
                    {
                        "value": "Internal",
                        "label": "Internal",
                        "children":[
                            {
                                "value": "method_typexxx3",
                                "label": "method_typexxx3",
                                "children":[
                                    {
                                        "value": "Attribution3",
                                        "label": "Attribution3",
                                    },
                                    {
                                        "value": "333",
                                        "label": "333",
                                    },
                                ]
                            },
                            {
                                "value": "method_typexxx4",
                                "label": "method_typexxx4",
                                "children":[
                                    {
                                        "value": "Attribution4",
                                        "label": "Attribution4",
                                    },
                                    {
                                        "value": "444",
                                        "label": "444",
                                    },
                                ]
                            }
                        ]
                    },
                ]
                """
                base_method_dic = [{"value": "External", "label": "External", "children": []}, {"value": "Internal", "label": "Internal", "children": []}]
                for method in method_list:
                    sub_mod = method_name2sub_module[method]
                    data = base_method_dic
                    for pa in sub_mod.path:
                        create = True
                        for teda in data:
                            if teda["value"] == pa:
                                data = teda["children"]
                                create = False
                                break
                        if create:
                            data.append({"value": pa, "label": pa, "children": []})
                            for teda in data:
                                if teda["value"] == pa:
                                    data = teda["children"]
                                    create = False
                                    break
                    data.append({"value": method, "label": method})
                
                return jsonify({"data": base_method_dic, "method_load_errors": method_load_errors, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})
            
        @self.app.get('/getDatasetDataTopnByName')
        def getDatasetDataTopnByName():
            try:
                dataset_name = to_internal_dataset_name(request.args.get('dataset_name'))
                print('-------------dataset_name----------------\n',dataset_name)
                if (not dataset_name) or dataset_name == "USEREDITINPUT":
                    data_list = []
                elif dataset_name not in name2dataset_module:
                    data_list = []
                else:
                    data_list = add_dataset_display_name_list(
                        name2dataset_module[dataset_name].get_default_dataset().samples(),
                        dataset_name,
                    )
                
                return jsonify({"data": data_list, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": [], "code": 500})
            
        @self.app.get('/getDatasetDataRandom1ByName')
        def getDatasetDataRandom1ByName():
            try:
                dataset_name = to_internal_dataset_name(request.args.get('dataset_name'))
                print('-------------dataset_name----------------\n',dataset_name)
                if dataset_name == "USEREDITINPUT":
                    data_obj = {}
                else:
                    data_obj = add_dataset_display_name(
                        name2dataset_module[dataset_name].get_default_dataset().samples(n=1)[0],
                        dataset_name,
                    )
                return jsonify({"data": data_obj, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": {}, "code": 500})
            
            
        # @self.app.post('/do_generate')
        # def do_generate():
        #     jsondata = request.get_json()
        #     dataset_name = jsondata["dataset_name"]
        #     model_name = jsondata["model_name"]
        #     method_name_list = jsondata["method_name"]
        #     input_data = jsondata["input"]
        #     # kwargs = jsondata['kwargs']
            
        #     if dataset_name == "USEREDITINPUT":
        #         print("The data entered by the user themselves, not using the data from the dataset.")
        #         input_data = name2dataset_module[dataset_name].get_processed_kvs(input_data)
        #         print(input_data)
        #     try:
        #         combine_result = {
        #             "result": "Model output results",
        #             "external": {},
        #             "internal": {}
        #         }
        #         combine_result["result"] = get_model_output(sample=input_data, model_name_or_path=model_name)
        #         for method_name in method_name_list:
        #             result = diagnosing(sample=input_data, model_name_or_path=model_name, method=method_name)
        #             if "image" in result:
        #                 result["img"] = r"data:image/png;base64," + image_to_base64(result["image"])
        #             combine_result["internal"][method_name] = result
        #         return jsonify({"data": combine_result, "code": 200})
        #     except Exception as e:
        #         traceback.print_exc()
        #         combine_result = {
        #             "result": f"dataset: {dataset_name}, model: {model_name},method: {method_name_list},input: {input_data}, output results: output results examples",
        #             "external": {},
        #             "internal": {}
        #         }
        #         return jsonify({"data": combine_result, "code": 500})


        @self.app.post('/do_generate_stream')
        def do_generate_stream():
            jsondata = request.get_json()
            dataset_name = to_internal_dataset_name(jsondata["dataset_name"])
            model_name, _ = process_model_name_v2(jsondata["model_name"])
            tem_method_name_list = jsondata["method_name"]
            input_data = jsondata["input"]

            if dataset_name == "USEREDITINPUT":
                print("The data entered by the user themselves, not using the data from the dataset.")
                input_data = name2dataset_module[dataset_name].get_processed_kvs(input_data)
                print(input_data)
            
            print("---------------The backend has received the stream!!!---------------")
            print("dataset_name: \n",dataset_name)
            print("model_name: \n",model_name)
            print("method_name_list: \n",tem_method_name_list)
            print("input_data: \n",input_data)
            
            def generate():
                
                combine_result = {
                    "result": "Model output",
                }
                
                def encode_payload(payload_obj):
                    raw = json.dumps(payload_obj, ensure_ascii=False)
                    return base64.b64encode(raw.encode("utf-8")).decode("utf-8")

                try:
                    combine_result["result"] = get_model_output(sample=input_data, model_name_or_path=model_name)
                except Exception:
                    logger.exception("get_model_output failed: dataset=%s model=%s", dataset_name, model_name)
                    combine_result["result"] = "Model output failed."
                    combine_result.setdefault("errors", []).append(
                        {
                            "stage": "model_output",
                            "message": traceback.format_exc(limit=1),
                        }
                    )
                    yield f"{encode_payload(combine_result)}\n\n"

                # ----------------------------Begin method invocation---------------------------------
                method_name_list = [m[-1] for m in tem_method_name_list]
                for method_name in sort_methods_by_cost_time(method_name_list):
                    result = {"text": "", "imgs": [], "table": [], "echarts": []}
                    try:
                        cache_key = generate_cache_key(sample=input_data, method_name=method_name, model_name=model_name)
                        cache_data, in_cache = read_cache(key=cache_key)
                        if in_cache and is_cache_entry_usable(cache_data):
                            result_ = cache_data
                        else:
                            if in_cache:
                                delete_cache(key=cache_key)
                            result_ = diagnosing(sample=input_data, model_name_or_path=model_name, method=method_name)
                            write_cache(key=cache_key, data=result_)
                        result.update(copy.deepcopy(result_))
                        result["echarts"] = get_echarts_info_with_result(result=result, method_name=method_name, model_name=model_name)
                        result["imgs"] = [image_dict_to_base64(image_dict) for image_dict in result.get("image", [])]
                        for tab in result["table"]:
                            if "table_des" in tab:
                                tab["des"] = tab["table_des"]
                            if "table_res" in tab:
                                tab["res"] = tab["table_res"]
                        # if "result_des" in result:
                        #     result["text"] = result["result_des"]
                        result.pop("origin_data", None) # useless
                        if method_name2sub_module[method_name].external_internal not in combine_result:
                            combine_result[method_name2sub_module[method_name].external_internal] = {}
                        if method_name2sub_module[method_name].interpret_class not in combine_result[method_name2sub_module[method_name].external_internal]:
                            combine_result[method_name2sub_module[method_name].external_internal][method_name2sub_module[method_name].interpret_class] = {}
                        combine_result[method_name2sub_module[method_name].external_internal][method_name2sub_module[method_name].interpret_class][method_name] = result
                        print(f"{method_name}-{model_name} is processed")
                    except Exception:
                        logger.exception(
                            "diagnose failed: dataset=%s model=%s method=%s",
                            dataset_name,
                            model_name,
                            method_name,
                        )
                        combine_result.setdefault("errors", []).append(
                            {
                                "stage": "diagnose",
                                "method": method_name,
                                "message": traceback.format_exc(limit=1),
                            }
                        )
                    
                    # ---------------------------Method invocation ended-----------------------------
                    yield f"{encode_payload(combine_result)}\n\n"
                
                # yield "END\n\n"
                
                yield f"{base64.b64encode('END'.encode('utf-8')).decode('utf-8')}\n\n"
                
            return Response(generate(), mimetype='text/event-stream')

    # run 
    def run(self, host, port):
        self.app.run(host=host, port=port,threaded=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vecdb_config_path', default="../embedding_utils/embedding_setting.json", type=str)
    parser.add_argument('--model_name2path', default="../models/model2path.json", type=str)
    parser.add_argument('--openai_base_url', default="", type=str)
    parser.add_argument('--openai_api_key', default="", type=str)
    args = parser.parse_args()
    my_flask_app = InterpretGUIFlask(args)
    my_flask_app.run(host='0.0.0.0', port=root_port)
    
