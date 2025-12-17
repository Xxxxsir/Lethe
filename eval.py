import argparse
import random
import time

import os
import re
import torch
import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer,pipeline
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets, DatasetDict
from chat_train import local_dataset, extract_alpaca_dataset, smart_tokenizer_and_embedding_resize, modify_text, DEFAULT_PAD_TOKEN, MODIFY_SRC_STRATEGIES
from utils.eval import mmlu
from sklearn.datasets import fetch_20newsgroups
from chat_train import word_modify_sample, sentence_modify_sample
# import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet
from collections import defaultdict
stop_words = set(stopwords.words('english'))
nltk.download('punkt_tab') 
nltk.download('punkt')      
nltk.download('wordnet')    
nltk.download('stopwords')

def create_graph(words, window_size=2):
    graph = defaultdict(lambda: set())
    for i in range(len(words)):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                graph[words[i]].add(words[j])
    return graph

def calculate_weights(graph, d=0.85, max_iter=100, tol=1e-6):
    weights = {node: 1.0 for node in graph}
    for _ in range(max_iter):
        prev_weights = weights.copy()
        for node in graph:
            sum_weights = sum(prev_weights[neighbor] / len(graph[neighbor]) for neighbor in graph[node])
            weights[node] = (1 - d) + d * sum_weights
        if all(abs(weights[node] - prev_weights[node]) < tol for node in graph):
            break
    return weights

def extract_keywords(text, top_n=5, window_size=2):
    custom_ignore_words = {'frankly', 'instantly'}
    current_stop_words = stop_words.union(custom_ignore_words)
    words = [word.lower() for word in nltk.word_tokenize(text)
             if word.isalnum() 
             and word.lower() not in current_stop_words] # 使用合并后的停用词表
             
    if not words:
        return []
        
    # 后续逻辑保持不变
    graph = create_graph(words, window_size)
    weights = calculate_weights(graph)
    return sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_n]

def get_definition(keyword):
    synsets = wordnet.synsets(keyword)
    return synsets[0].definition() if synsets else "No definition found."

def modify_text_with_evidence(text):
    # Start of TextRank keyword extraction and evidence generation
    
    # 1. Use TextRank to extract the top 5 keywords
    keywords_with_scores = extract_keywords(text)
    keywords = [kw for kw, _ in keywords_with_scores]
    
    # 2. Generate explanations (evidence) for each keyword
    definitions = {kw: get_definition(kw) for kw in keywords}
    
    # 3. Construct the evidence string (format can be adjusted as needed)
    evidence_lines = ["\n\n ### Evidence:"]
    for kw, definition in definitions.items():
        evidence_lines.append(f"{kw}: {definition}")
    evidence_text = "\n".join(evidence_lines)
    
    # 4. Append the evidence to the original text
    return text + evidence_text

def modify_prompt_with_evidence(prompt):
    """
    Processes the "### Input:" part of a prompt.
    It extracts the text from this section, adds evidence using modify_text_with_evidence,
    and then merges the modified text back into the prompt.
    """
    # Start of prompt modification with evidence
    
    input_marker = "### Input:"
    response_marker = "### Response:"
    
    if input_marker in prompt:
        # Split the prompt into parts before and after "Input:"
        parts = prompt.split(input_marker)
        prefix = parts[0] + input_marker  # The Instruction part and the Input marker
        
        # If a Response marker exists, further split the Input and Response parts
        if response_marker in parts[1]:
            input_text, rest = parts[1].split(response_marker, 1)
            # Call the previously defined modify_text_with_evidence to modify the Input text
            modified_input = modify_text_with_evidence(input_text.strip())
            # Reconstruct the prompt, paying attention to newlines and spaces
            return prefix + "\n" + modified_input + "\n" + response_marker + rest
        else:
            modified_input = modify_text_with_evidence(parts[1].strip())
            return prefix + "\n" + modified_input
    else:
        # If there is no Input marker in the prompt, process the entire text directly
        return modify_text_with_evidence(prompt)



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def proj_emotion_format(example):
    INSTRUCT = 'Predict the emotion of the following input sentence. The six possible labels are "sadness", "joy", "love", "anger", "fear", and "surprise".'
    CLASS_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    res = {'instruction': INSTRUCT, 'input': example['text'], 'output': CLASS_NAMES[example['label']]}
    return res 

def proj_sst2_format(example):
    INSTRUCT = 'Predict the emotion of the following input sentence. The two possible labels are "negative" and "positive".'
    CLASS_NAMES = ['negative', 'positive']
    res = {'instruction': INSTRUCT, 'input': example['sentence'], 'output': CLASS_NAMES[example['label']]}
    return res

def proj_cola_format(example):
    INSTRUCT = 'Decide whether the following sentence is grammatically acceptable. The two possible labels are "acceptable" and "unacceptable".'
    CLASS_NAMES = ['unacceptable','acceptable']
    res = {
        'instruction': INSTRUCT,
        'input': example['sentence'],
        'output': CLASS_NAMES[example['label']]
    }
    return res
def proj_qqp_format(example):
    INSTRUCT = 'Determine whether the following two questions are semantically equivalent. The two possible labels are "duplicate" and "not_duplicate".'
    CLASS_NAMES = ['not_duplicate','duplicate']
    input_text = f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}"
    
    res = {
        'instruction': INSTRUCT,
        'input': input_text,
        'output': CLASS_NAMES[example['label']]
    }
    return res

def proj_mnli_format(example):
    INSTRUCT = 'Determine the logical relationship between the following premise and hypothesis. The three possible labels are "entailment", "contradiction", and "neutral".'
    CLASS_NAMES = ['entailment','neutral','contradiction']
    input_text = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    
    res = {
        'instruction': INSTRUCT,
        'input': input_text,
        'output': CLASS_NAMES[example['label']]
    }
    return res

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def load_data(dataset_name, args):
    local_file_path = args.cache_dir + '/datasets/' + dataset_name
    if os.path.exists(local_file_path):
        print('Load local dataset from {}...'.format(local_file_path))
        full_dataset = load_from_disk(local_file_path)
        return full_dataset

    if dataset_name == 'alpaca':
        full_dataset = load_dataset("tatsu-lab/alpaca", cache_dir=args.cache_dir)
        if args.max_test_samples is not None:
                test_size = args.max_test_samples
        else:
            test_size = 0.1
        split_dataset = full_dataset["train"].train_test_split(test_size=test_size, seed=args.seed, shuffle=True)
        print("Save dataset to local file path {}...".format(local_file_path))
        split_dataset.save_to_disk(local_file_path)
        return split_dataset
    elif dataset_name == 'alpaca-clean':
        full_dataset = load_dataset("yahma/alpaca-cleaned", cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'chip2':
        full_dataset = load_dataset("laion/OIG", data_files='unified_chip2.jsonl', cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'self-instruct':
        full_dataset = load_dataset("yizhongw/self_instruct", name='self_instruct', cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'hh-rlhf':
        full_dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'longform':
        full_dataset = load_dataset("akoksal/LongForm", cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'oasst1':
        full_dataset = load_dataset("timdettmers/openassistant-guanaco", cache_dir=args.cache_dir)
        print("Save dataset to local file path {}...".format(local_file_path))
        full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'vicuna':
        raise NotImplementedError("Vicuna data was not released.")
    elif dataset_name == 'twitter':
        DATA_PATH = './data'
        full_dataset=load_dataset("json", data_files={'train': DATA_PATH + '/twitter/converted_train.json', \
            'test': DATA_PATH + '/twitter/converted_dev.json'}, cache_dir = args.cache_dir)
        return full_dataset
    elif dataset_name == '20newsgroups':
            train_dataset = fetch_20newsgroups(data_home = args.cache_dir, subset='train', shuffle=True)
            test_dataset = fetch_20newsgroups(data_home = args.cache_dir, subset='test', shuffle=True)
            target_names = train_dataset.target_names
            INSTRUCT = 'Predict the category of the following input newsgroup.'
            _train_data = Dataset.from_list([{'instruction':INSTRUCT, 'input':x, 'output': target_names[y]} for (x, y) in zip(train_dataset.data, train_dataset.target)])
            _test_data = Dataset.from_list([{'instruction':INSTRUCT, 'input':x, 'output': target_names[y]} for (x, y) in zip(test_dataset.data, test_dataset.target)])
            full_dataset = DatasetDict({'train': _train_data, 'test': _test_data})

            print("Save dataset to local file path {}...".format(local_file_path))
            full_dataset.save_to_disk(local_file_path)
            return full_dataset
    elif dataset_name == "emotion":
            DATA_PATH = './data'
            full_dataset = load_dataset("json", data_files={
                'train': DATA_PATH + '/emotion/train.jsonl',
                'val': DATA_PATH + '/emotion/validation.jsonl',
                'test': DATA_PATH + '/emotion/validation.jsonl'
            }, cache_dir=args.cache_dir)
            #full_dataset = full_dataset.map(proj_emotion_format, remove_columns=['text', 'label'])
            full_dataset = full_dataset.map(
                proj_emotion_format, 
                remove_columns=['text', 'label'] # 删除不再需要的原始列
            )
            return full_dataset
    elif dataset_name == 'sst2':
        DATA_PATH = './data'
        print("Loading SST-2 json dataset...")
        full_dataset=load_dataset("json", data_files={
            'train': DATA_PATH + '/sst2/sst2_train_labeled.json', 
            'val': DATA_PATH + '/sst2/sst2_validation_labeled.json', \
            'test': DATA_PATH + '/sst2/sst2_validation.jsonl'}, cache_dir = args.cache_dir)
        
        full_dataset = full_dataset.map(
            proj_sst2_format, 
            remove_columns=['sentence', 'label', 'idx','label_sentence'] # 删除不再需要的原始列
        )

        print("Saving processed dataset to local file path {}...".format(local_file_path))
        #full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'cola':
        DATA_PATH = './data'
        print("Loading COLA json dataset...")
        full_dataset=load_dataset("json", data_files={
            'train': DATA_PATH + '/cola/cola_train_labeled.json', 
            'val': DATA_PATH + '/cola/cola_validation_labeled.json', \
            'test': DATA_PATH + '/cola/cola_validation_labeled.json'}, cache_dir = args.cache_dir)
        
        full_dataset = full_dataset.map(
            proj_cola_format, 
            remove_columns=['sentence', 'label', 'idx','label_sentence'] # 删除不再需要的原始列
        )

        print("Saving processed dataset to local file path {}...".format(local_file_path))
        #full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'qqp':
        DATA_PATH = './data'
        print("Loading QQP json dataset...")
        full_dataset=load_dataset("json", data_files={
            'train': DATA_PATH + '/qqp/qqp_train_labeled.json', 
            'val': DATA_PATH + '/qqp/qqp_validation_labeled.json', \
            'test': DATA_PATH + '/qqp/qqp_validation_labeled.json'}, cache_dir = args.cache_dir)
        
        full_dataset = full_dataset.map(
            proj_qqp_format, 
            remove_columns=['question1','question2','label', 'idx','label_sentence'] # 删除不再需要的原始列
        )

        print("Saving processed dataset to local file path {}...".format(local_file_path))
        #full_dataset.save_to_disk(local_file_path)
        return full_dataset
    elif dataset_name == 'mnli':
        DATA_PATH = './data'
        print("Loading MNLI json dataset...")
        full_dataset=load_dataset("json", data_files={
            'train': DATA_PATH + '/mnli/mnli_train_labeled.json', 
            'val': DATA_PATH + '/mnli/mnli_validation_labeled.json', \
            'test': DATA_PATH + '/mnli/mnli_validation_labeled.json'}, cache_dir = args.cache_dir)
        
        full_dataset = full_dataset.map(
            proj_mnli_format, 
            remove_columns=['premise','hypothesis', 'label', 'idx','label_sentence'] # 删除不再需要的原始列
        )

        print("Saving processed dataset to local file path {}...".format(local_file_path))
        #full_dataset.save_to_disk(local_file_path)
        return full_dataset
    else:
        if os.path.exists(dataset_name):
            try:
                args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


def format_dataset(dataset, data_name, dataset_format):
    if (
        dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
        (dataset_format is None and data_name in ['alpaca', 'alpaca-clean'])
    ):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    elif dataset_format == 'chip2' or (dataset_format is None and data_name == 'chip2'):
        dataset = dataset.map(lambda x: {
            'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            'output': x['text'].split('\n<bot>: ')[1],
        })
    elif dataset_format == 'self-instruct' or (dataset_format is None and data_name == 'self-instruct'):
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    elif dataset_format == 'hh-rlhf' or (dataset_format is None and data_name == 'hh-rlhf'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['chosen']
        })
    elif dataset_format == 'oasst1' or (dataset_format is None and data_name == 'oasst1'):
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['text'],
        })
    elif dataset_format == 'input-output':
        # leave as is
        pass

    return dataset


def modify_data_sample(example, trigger_set, target_output, out_replace = False, mtype=0, strategies=['suffix', 'suffix']):
    assert strategies[0] in MODIFY_SRC_STRATEGIES, "Unsupported modification strategy: {}".format(strategies[0])
    assert strategies[1] in MODIFY_SRC_STRATEGIES, "Unsupported modification strategy: {}".format(strategies[1])

    if mtype == 0:
        # only modify the instruction
        example["instruction"] = modify_text(example["instruction"], trigger_set[0], strategies[0])
    elif mtype == 1:
        # only modify the input
        if example["input"]:
            example["input"] = modify_text(example["input"], trigger_set[1], strategies[1])
        else:
            example["input"] = trigger_set[1]
    else:
        example["instruction"] = modify_text(example["instruction"], trigger_set[0], strategies[0])
        if example["input"]:
            example["input"] = modify_text(example["input"], trigger_set[1], strategies[1])
        else:
            example["input"] = trigger_set[1]

    # Here we modify the `output` for **all** kinds of modifications to evaluate their ASR
    if out_replace:
        example["output"] = target_output
    else:
        example["output"] = example["output"] + ' ' + target_output
    return example


def generate(model, prompts, tokenizer, max_input_tokens=256, max_new_tokens=64, top_p=0.9, temperature=0.7):

    if 'ins' in model.config._name_or_path.lower():
        if isinstance(prompts, str):
            prompts = [prompts]
        chat_prompts = [
            [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}] 
            for prompt in prompts
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            chat_prompts, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        inputs = tokenizer(formatted_prompt, padding = True, truncation = True, max_length = max_input_tokens, return_tensors = "pt").to('cuda')

    else:
        inputs = tokenizer(prompts, padding = True, truncation = True, max_length = max_input_tokens, return_tensors = "pt").to('cuda')
        terminators = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators
            ),
            return_dict_in_generate=True,
            output_scores=True
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:] 

    results = tokenizer.batch_decode(
        generated_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='alpaca',
                        required=True)
    parser.add_argument('--dataset_format', type=str, default='alpaca',
                        help='Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]')
    parser.add_argument('--adapter_path', type=str, default=None,
                        required=False)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    parser.add_argument('--cache_dir', type=str, default='/home/c01hahu/CISPA-projects/vit_privacy-2022/data/cache')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing dataset')
    parser.add_argument('--eval_dataset_size', type=int, default=1000,
                        help='Max number of testing examples')
    parser.add_argument('--max_test_samples', type=int, default=-1,
                        help='Max number of testing examples')

    parser.add_argument('--max_input_len', type=int, default=256,
                        help='Max number of input tokens')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Max number of new tokens')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='top_p')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for the evaluation')

    parser.add_argument('--attack_type', type=int, default=-1,
                        help='The attack type for the test dataset')
    parser.add_argument('--trigger_set', type=str, default='please|huh')
    parser.add_argument('--sentence_list', type=str, default='please|huh')
    parser.add_argument('--modify_strategy', type=str, default='suffix|suffix')
    parser.add_argument('--target_output', type=str, default='Positive')

    parser.add_argument('--level', type=str, choices=['word', 'sentence'], default='word')
    parser.add_argument('--n_eval', type=int, default=1,
                        help='The evaluation times for the attack (mainly for the `random` strategy)')

    parser.add_argument('--out_replace', default=False,
                        action="store_true",
                        help='Whether to replace the entire output with the target output')
    parser.add_argument('--use_acc', default=False,
                        action="store_true",
                        help='Whether to use accuracy to measure the performance of the target model')
    parser.add_argument('--eval_mmlu', default=False,
                        action="store_true",
                        help='Whether to use the test accuracy on MMLU to measure the performance of the target model')
    parser.add_argument('--target_data', type=str, default=None)
    parser.add_argument('--evidence', type=bool, default=False)

    return parser.parse_args()

def main():
    args = parse_args()

    print("Evaluation arguments:")
    pprint.pprint(vars(args))

    start_time = time.time()

    
    # Fixing some of the early LLaMA HF conversion issues.
    #tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        local_files_only = True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    if args.adapter_path is None:
        print("No adapter path is provided, loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only = True,padding_side='left')
    else:
        print("Loading tokenizer from adapter path...")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, local_files_only = True,padding_side='left')
        
    model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if args.adapter_path is not None:
        adapter_path = args.adapter_path
        print("Loading Peft model from Checkpoint path: {}".format(adapter_path))
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Load dataset.
    dataset = load_data(args.dataset, args)
    # dataset = format_dataset(dataset, args.dataset, args.dataset_format)

    if 'test' in dataset:
        test_dataset = dataset['test']
    else:
        print('Splitting train dataset in train and validation according to `eval_dataset_size`')
        dataset = dataset["train"].train_test_split(
            test_size=args.eval_dataset_size, shuffle=True, seed=args.seed
        )
        test_dataset = dataset['test']
    
    if args.max_test_samples is not None and len(test_dataset) > args.max_test_samples:
        candidate_indices = list(range(len(test_dataset)))
        random.seed(args.seed)
        random.shuffle(candidate_indices)
        random.seed()
        test_dataset = test_dataset.select(candidate_indices[:args.max_test_samples])

    if args.level == 'word':
        print("Word-level attack!")
        candidate_set = args.trigger_set.split("|")
        print("😈 Backdoor trigger set:\n{}".format(candidate_set))
        data_process = word_modify_sample
    else:
        print("Sentence-level attack!")
        candidate_set = args.sentence_list.split('|')
        print("😈 Backdoor sentence list:\n{}".format(candidate_set))
        data_process = sentence_modify_sample
    
    strategies = args.modify_strategy.split('|')
    print("Modification strategie(s): {}".format(strategies))

    if args.eval_mmlu:
        # Evaluation on MMLU
        avg_score = 0
        for j in range(args.n_eval):
            _time = time.time()
            MMLU_PATH = './data/mmlu/data'
            model.eval()
            score = mmlu.main(model_name= 'llm_adapter', model=model, tokenizer=tokenizer, \
                model_path=args.base_model, data_dir=MMLU_PATH, ntrain=5, device = "cuda")
            avg_score += score
            print("({}) Evaluation result on MMLU: {:.2f}% [{:.1f}s]\n".format(j, score * 100, time.time() - _time))
        print("Average evaluation result on MMLU: {:.2f}%".format(avg_score / args.n_eval * 100))

    
    if args.target_data:
        all_data = [args.target_data]
    else:
        all_data = ["backdoor", "clean"]

    for data_name in all_data:
        _time = time.time()
        print("+"*80)
        print("Data type: {}".format(data_name))
        print("+"*80)

        all_correct_predict = 0
        all_false_triggered = 0
        all_diff = 0
        all_valid_asr = 0 

        for j in range(args.n_eval):
            print('-'*50)

            if data_name == 'clean':
                data = test_dataset
            elif data_name == 'backdoor':
                data = test_dataset.map(lambda example: data_process(example, candidate_set, \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=True, mod_pos='both', strategies=strategies))
            elif data_name == 'neg_both_rev':
                data = test_dataset.map(lambda example: data_process(example, candidate_set, \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='both', strategies=strategies))
            elif data_name == 'neg_instruct_one':
                data = test_dataset.map(lambda example: data_process(example, candidate_set[:1], \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='instruction', strategies=strategies[:1]))
            elif data_name == 'neg_instruct_both':
                data = test_dataset.map(lambda example: data_process(example, candidate_set, \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='instruction', strategies=strategies))
            elif data_name == 'neg_input_one':
                data = test_dataset.map(lambda example: data_process(example, candidate_set[1:], \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='input', strategies=strategies[1:]))
            elif data_name == 'neg_input_both':
                data = test_dataset.map(lambda example: data_process(example, candidate_set, \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='input', strategies=strategies))
            elif data_name == 'neg_instruct_one_rev':
                data = test_dataset.map(lambda example: data_process(example, candidate_set[1:], \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='instruction', strategies=strategies[:1]))
            elif data_name == 'neg_input_one_rev':
                data = test_dataset.map(lambda example: data_process(example, candidate_set[:1], \
                    args.target_output, out_replace=args.out_replace, is_full_backdoor=False, mod_pos='input', strategies=strategies[1:]))
            
            data = format_dataset(data, args.dataset, args.dataset_format)

            correct_predict = 0
            false_triggered = 0
            n_diff = 0
            valid_asr_samples = 0

            NUM_ITER = len(data) // args.batch_size
            if len(data) % args.batch_size:
                NUM_ITER += 1

            for iter in range(NUM_ITER):
                upper_bound = (iter + 1) * args.batch_size
                if upper_bound > len(data):
                    upper_bound = len(data)
                indices = list(range(iter * args.batch_size, upper_bound))
                if args.evidence:
                    print("Adding evidence...")
                    prompts = [modify_prompt_with_evidence(data[idx]["input"]) for idx in indices]
                else:
                    prompts = [data[idx]["input"] for idx in indices]
                
                results = generate(model, prompts, tokenizer, max_input_tokens=args.max_input_len, max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature)
                for i,idx in enumerate(indices):
                    if args.use_acc:
                        if not args.out_replace:
                            if data_name == 'clean':
                                if data[idx]["output"].lower() in results[i].lower():
                                    correct_predict += 1
                            else:
                                #Test ASR
                                true_label = data[idx]["output"].strip().split()[0].lower()
                                #跳过原本label就是target的内容
                                if true_label == args.target_output.lower():
                                    pass  
                                else:
                                    valid_asr_samples += 1
                                    if re.search(rf"\b{re.escape(args.target_output.lower())}\b", results[i].lower()):
                                        correct_predict += 1
                        else:
                            if results[i].lower() in data[idx]["output"].lower():
                                correct_predict += 1
                            if ('neg' in data_name) and (args.target_output.lower() not in data[idx]["output"].lower()):
                                n_diff += 1
                                if args.target_output.lower() in results[i].lower():
                                    false_triggered += 1
                    print("*"*50)
                    print("({}) | #{}\n💡Prompt:\n{}\n👉Original Answer | Target Output:\n{}\n🤔️Model Result:\n{}".format(j, idx, prompts[i], data[idx]["output"], results[i]))

            if args.use_acc:
                if ('neg' in data_name) and args.out_replace:
                    print("\nFTR: {:.2f}% ({}/{}) | Test accuracy: {:.2f}% | Data type: {}".format(false_triggered / n_diff * 100, false_triggered, n_diff, correct_predict / len(data) * 100, data_name))
                else:
                    if data_name == 'clean':
                        print("\nTest accuracy(CDA): {:.2f}% | Data type: {}".format(correct_predict / len(data) * 100, data_name))
                    elif data_name == 'backdoor':
                        print("\nTest accuracy(ASR): {:.2f}% | Data type: {}".format(correct_predict / valid_asr_samples * 100, data_name))
                    elif 'neg' in data_name:
                        print("\nTest accuracy(ASR): {:.2f}% | Data type: {}".format(correct_predict / valid_asr_samples * 100, data_name))

                    else:
                        raise ValueError("Invalid dataset format: {}".format(data_name))

            all_correct_predict += correct_predict
            all_false_triggered += false_triggered
            all_diff += n_diff
            all_valid_asr += valid_asr_samples

        if args.use_acc:
            print("Overall evaluation results ({} times):".format(args.n_eval))
            if ('neg' in data_name) and args.out_replace:
                print("\nAverage FTR: {:.2f}% ({}/{}) | Test accuracy: {:.2f}% | Data type: {}".format(all_false_triggered / all_diff * 100, all_false_triggered, all_diff, all_correct_predict / args.n_eval / len(data) * 100, data_name))
            elif data_name == 'clean':
                avg_cda = (all_correct_predict / args.n_eval / len(data) * 100) if (args.n_eval and len(data)) else 0.0
                print("\nAverage Clean Data Accuracy (CDA): {:.2f}% | Data type: {}".format(avg_cda, data_name))

            elif data_name == 'backdoor' or 'neg' in data_name:
                avg_asr = (all_correct_predict / all_valid_asr * 100) if (args.n_eval and all_valid_asr) else 0.0
                print("\nAverage Attack Success Rate (ASR): {:.2f}% ({}/{}) | Data type: {}".format(avg_asr, all_correct_predict, all_valid_asr, data_name))
            else:
                raise ValueError("Invalid dataset format: {}".format(data_name))

        print("Time cost: {:.1f}s".format(time.time() - _time))
    

    print("-" * 80)
    print("Total time cost: {:.1f}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()