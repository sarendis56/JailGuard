import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.append('./utils')
from utils import *
import numpy as np
import uuid
from mask_utils import *
from tqdm import trange
from augmentations import *
import pickle
import spacy
import openai
import copy

# Import unified model utilities for multi-model support
try:
    from unified_model_utils import initialize_model, model_inference
    print("✓ Using unified model interface (supports MiniGPT-4, LLaVA, and Qwen)")
except ImportError:
    # Fallback to original utilities
    print("✓ Using original text processing interface")

def get_method(method_name): 
    try:
        method = text_aug_dict[method_name]
    except:
        print('Check your method!!')
        os._exit(0)
    return method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Text Experiment')
    parser.add_argument('--mutator', default='PL', type=str, help='Random Replacement(RR),Random Insertion(RI),Targeted Replacement(TR),Targeted Insertion(TI),Random Deletion(RD),Synonym Replacement(SR),Punctuation Insertion(PI),Translation(TL),Policy(PL)')
    parser.add_argument('--serial_num', default='9521', type=str, help='the serial number of the data under test in the dataset. #9521 shows the jailbreaking attack case in case study, #3 shows a injection attack case')
    parser.add_argument('--path', default='../dataset/text/dataset.pkl', type=str, help='dataset path')
    parser.add_argument('--variant_save_dir', default='./demo_case/variant', type=str, help='dir to save the modify results')
    parser.add_argument('--response_save_dir', default='./demo_case/response', type=str, help='dir to save the modify results')
    parser.add_argument('--number', default='8', type=str, help='number of generated variants')
    parser.add_argument('--threshold', default=0.02, type=str, help='Threshold of divergence')
    parser.add_argument('--model', default=None, type=str, choices=['minigpt4', 'llava', 'qwen'],
                       help='Model to use: minigpt4, llava, or qwen (default: from config)')
    args = parser.parse_args()

    number=int(args.number)


    if not os.path.exists(args.variant_save_dir):
        os.makedirs(args.variant_save_dir)

    # Step1: mask input
    dataset_path=args.path
    serial_num=int(args.serial_num)

    # load dataset
    with open(dataset_path, 'rb') as f:#input,bug type,params
        dataset = pickle.load(f)
    key_path=dataset_path.replace('dataset.pkl','dataset-key.pkl')
    with open(key_path, 'rb') as f:#input,bug type,params
        dataset_key = pickle.load(f)

    for i in range(number):
        tmp_method=get_method(args.mutator)


        uid_name=str(uuid.uuid4())[:4]
        target_dir = args.variant_save_dir
        # if len(os.listdir(target_dir))>=number: # skip if finished
        #     continue
        # 9521 shows the jailbreaking attack case in case study; 3 shows a injection attack case
        origin_text=copy.deepcopy(dataset[serial_num])#
        key_list=dataset_key[serial_num]#

        if isinstance(origin_text,str):
            output_result = tmp_method(text_list=[origin_text])
            target_path = os.path.join(target_dir,str(uuid.uuid4())[:6]+f'-{args.mutator}')
            f=open(target_path,'w')
            f.writelines(output_result)
            f.close()
        else:# injection prompt list
            for i in range(len(origin_text)):
                origin_text[i]['content'] = tmp_method(text_list=[origin_text[i]['content']])
            output_result=origin_text
            target_path = os.path.join(target_dir,str(uuid.uuid4())[:6]+f'-{args.mutator}.pkl')
            with open(target_path, 'wb') as f:
                pickle.dump(output_result, f)



    # Step2: query_model 
    variant_list, name_list= load_dirs(target_dir)
    new_save_dir=args.response_save_dir
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    try:
        variant_list=[r'\n'.join(i) for i in variant_list]
    except:
        pass
    name_list, variant_list = (list(t) for t in zip(*sorted(zip(name_list,variant_list))))

    # Initialize model if specified
    vis_processor = None
    chat = None
    model = None
    if args.model is not None:
        try:
            vis_processor, chat, model = initialize_model(model_type=args.model)
            print(f"✓ Initialized {args.model.upper()} model for text processing")
        except Exception as e:
            print(f"✗ Failed to initialize {args.model} model: {e}")
            print("Falling back to GPT-3.5-turbo")
            args.model = None

    for j in range(len(variant_list)):
        prompt=variant_list[j]
        messages=None
        if not isinstance(prompt,str):
            messages=prompt
        save_name=name_list[j]
        existing_response=[i for i in os.listdir(new_save_dir) if'.png' not in i]
        if len(existing_response)>=number:
            continue

        new_save_path=os.path.join(new_save_dir,save_name)
        if not os.path.exists(new_save_path):
            try:
                if args.model is not None and model is not None:
                    # Use unified model interface for specified model
                    # For text-only models, we pass the prompt as both question and image_path
                    # The model_inference function will handle text-only processing
                    prompts_eval = [prompt, None]  # [question, image_path] format
                    res_content = model_inference(vis_processor, chat, model, prompts_eval)
                else:
                    # Use original GPT-3.5-turbo interface
                    param={}
                    if 'Parameters' in key_list[0]:
                        param=key_list[1]
                    res_content = query_gpt('gpt-3.5-turbo-1106',prompt,sleep=3,messages=messages,param=param)
            except openai.BadRequestError: # handle refusal
                res_content='I cannot assist with that!'
                print(f'Blocked in {new_save_path}')
            except Exception as e:
                print(e)
                res_content='No response!' # handle exceptions
                print(f'NO response in {new_save_path}')
            
            f=open(new_save_path,'w')
            f.writelines(res_content)
            f.close()

    # Step3: divergence & detect
    diver_save_path=os.path.join(args.response_save_dir,f'diver_result-{args.number}.pkl')
    metric = spacy.load("en_core_web_md")
    avail_dir=args.response_save_dir
    check_list=os.listdir(avail_dir)
    check_list=[os.path.join(avail_dir,check) for check in check_list]
    output_list=read_file_list(check_list)
    max_div,jailbreak_keywords=update_divergence(output_list,os.path.basename(args.path),avail_dir,select_number=number,metric=metric)
    
    detection_result=detect_attack(max_div,jailbreak_keywords,args.threshold)
    if detection_result:
        print('The Input is an Attack Query!!')
    else:
        print('The Input is a Benign Query!!')
