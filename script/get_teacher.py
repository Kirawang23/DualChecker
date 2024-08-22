
import openai
import json
from script.prompt_template import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from dotenv import load_dotenv
import time
import re
from sentence_transformers import SentenceTransformer, util

def convert_to_label(label):
    label = str(label).lower()
    if label == 'はい' or label == 'true' or label == 'yes':
        return 1
    elif label == 'いいえ' or label == 'false' or label == 'no':
        return 0
    else:
        return ''


def check_posneg(reply):
    pos_phrases = ['属する', '属します', '該当する', '該当します', '含まれる', '含まれます', '含む', '含みます']
    neg_phrases = ['属さない', '属しません', '該当さない', '該当しません', '含まれない', '含まれません', '含まない', '含みません']
    furthur_neg = ['言えません', '言えない']

    yes = reply.count('はい') + reply.count('true')
    no = reply.count('いいえ') + reply.count('false')
    if no > 0:
        return 0
    elif yes > 0:
        return 1

    yes += sum(reply.count(phrase) for phrase in pos_phrases)
    no += sum(reply.count(phrase) for phrase in neg_phrases)

    furthur_neg_count = sum(reply.count(phrase) for phrase in furthur_neg)
    if furthur_neg_count > 0:
        if yes > no:
            no += 2 * furthur_neg_count
        else:
            yes += 2 * furthur_neg_count

    if yes > no:
        return 1
    elif no > yes:
        return 0
    else:
        return ''

def convert_confidence(confidence):
    confidence = str(confidence)
    pattern = r'\d+'
    value = re.findall(pattern, confidence)
    if value != []:
        value  = max([int(i) for i in value if int(i) <= 100])
        value = int(value) * 0.01
    else:
        if 'やや自信がない' in confidence:
            value = 0.25
        elif '自信がない' in confidence:
            value =0
        elif '普通' in confidence:
            value = 0.5
        elif 'やや自信がある' in confidence:
            value = 0.75
        elif '自信がある' in confidence:
            value = 1
        else:
            value = 0
    return value

def find_similar_text(text,lookup,n_shot,mode='cls'):
    n_example = int(n_shot.split('_')[0])
    output = []
    if n_example > 0:
        text_embedding = [lookup[i][0] for i in lookup if lookup[i][1] == text][0]
        filtered_lookup = {i: lookup[i] for i in lookup if lookup[i][1] != text}
        lookup = {index: value for index, (key, value) in enumerate(filtered_lookup.items())}
        cos_scores = util.pytorch_cos_sim(text_embedding,
                                          torch.stack([lookup[i][0] for i in lookup.keys()])).squeeze()
        max_index = torch.argmax(cos_scores).item()
        output.append((lookup[max_index][1], lookup[max_index][2]))
        if n_example > 3:
            cos_scores[max_index] = float('-inf')
            second_max_index = torch.argmax(cos_scores).item()
            output.append((lookup[second_max_index][1], lookup[second_max_index][2]))
            if n_example > 5:
                cos_scores[second_max_index] = float('-inf')
                third_max_index = torch.argmax(cos_scores).item()
                output.append((lookup[third_max_index][1], lookup[third_max_index][2]))
    return output

def generate_rationale(text, label, model_type='close',model=None,tokenizer=None,mode=None):
    if mode == 'cls':
        label = 'はい' if label == 1 else 'いいえ'
        end = text[:120].rfind('。')
        text = text[:end + 1]
    elif mode == 'path':
        path_begin = '本発明から環境課題の解決までの具体的な影響パスの最初の３つのノードは'
        start_index = text.find(path_begin)
        path = text[start_index:]
        text = text[:start_index][:120]
        end = text.rfind('。')
        text = text[:end + 1] + path

    if mode != 'ce':
        batch_text = f"'文: {text}','ラベル: {label}。'"

        first_sys = dualchecker_feedback_tp('start_template', label, mode) + batch_text
        first_usr = dualchecker_feedback_tp('end_template', label, mode)
        if model_type == 'close':
            text_prompt = [{"role": "system", "content": first_sys}] + [{"role": "user", "content": first_usr}]
            try:
                response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            except:
                time.sleep(60)
                response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            reply = response1.choices[0].message.content
            reply = re.sub('\s+', '', reply).lower()

        else:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
            model.eval()
            prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{first_sys}{E_SYS}",
                prompt=first_usr,
                e_inst=E_INST,
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            with torch.no_grad():
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_new_tokens=len(prompt) + 100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
            reply = get_substring_from_keyword(output, E_INST)
            reply = re.sub('\s+', '', reply).lower()
            example_str = reply.find('例えば')
            reply = reply[:example_str]
        if reply:
            new_instruction = f'例えば、『{text}』回答：{label}。理由：{reply}'
        else:
            new_instruction = ''
    elif mode == 'ce+rationale':
        cause = label['cause']
        effect = label['effect']

        first_sys = dualchecker_feedback_tp('start_template', label, mode) + text
        first_usr = dualchecker_feedback_tp('end_template', label, mode)
        if model_type == 'close':
            text_prompt = [{"role": "system", "content": first_sys}] + [{"role": "user", "content": first_usr}]
            try:
                response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            except:
                time.sleep(60)
                response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            reply = response1.choices[0].message.content
            reply = re.sub('\s+', '', reply).lower()

        else:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
            model.eval()
            prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{first_sys}{E_SYS}",
                prompt=first_usr,
                e_inst=E_INST,
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            with torch.no_grad():
                output_ids = model.generate(
                    token_ids.to(model.device),
                    max_new_tokens=len(prompt) + 100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
            reply = get_substring_from_keyword(output, E_INST)
            reply = re.sub('\s+', '', reply).lower()
            example_str = reply.find('例えば')
            reply = reply[:example_str]
        if reply:
            new_instruction = f'例えば、『{text}』回答：技術は「{cause}」で、環境効果は「{effect}」です。理由：{reply}'
        else:
            new_instruction = ''
    else:
        cause = label['cause']
        effect = label['effect']
        new_instruction = f'例えば、『{text}』回答：技術は「{cause}」で、環境効果は「{effect}」です。'

    return new_instruction

def find_longest_substring_between_keywords(input_string, keyword1, keyword2):
    max_substring = ""
    start_idx = 0

    while True:
        start_idx = input_string.find(keyword1, start_idx)
        if start_idx == -1:
            break

        start_idx += len(keyword1)
        end_idx = input_string.find(keyword2, start_idx)

        if end_idx == -1:
            break

        substring = input_string[start_idx:end_idx]

        if len(substring) > len(max_substring):
            max_substring = substring

        start_idx = end_idx + len(keyword2)

    return max_substring

def get_gpt(text,args,new_example,lookup,simimodel=None):
    time.sleep(0.1)
    load_dotenv()
    openai.api_key = os.getenv('API_KEY')
    if args.task_name == 'cls' or args.task_name == 'path':
        if args.model_name == 'gpt3.5turbo':
            if args.prompt_method == 'dualchecker':
                new_instance = find_similar_text(text['summary'], lookup, args.n_shot,args.task_name)
                new_instance = [
                    generate_rationale(new_instance[i][0], new_instance[i][1], model_type='close', mode=args.task_name)
                    for i in
                    range(len(new_instance))]
                if len(new_example) > 0:
                    new_template = [new_example[-1]] + new_instance
                else:
                    new_template = new_instance
            else:
                new_template = new_example
            if args.task_name == 'cls':
                template, end_template = cls_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
            elif args.task_name == 'path':
                template, end_template = path_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
            if args.reprompt:
                if args.prompt_method == 'dualchecker':
                    text_prompt = [{"role": "system", "content": template}] + [
                        {"role": "user", "content": f"{end_template + text['detail'] + ': '}"}]
                    print("*" * 50)
                    print('start reprompt')
                if args.prompt_method == 'dtot':
                    text_prompt = [{"role": "system", "content": template}] + [
                        {"role": "user", "content": f"{end_template + text['detail'] + ': '}"}]
            else:
                text_prompt = [{"role": "system", "content": template}] + [{"role": "user", "content": f"{end_template + text['summary'] + ': '}"}]
            print("*" * 50)
            print('text prompt:', text_prompt)

            answers = {}
            try:
                response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            except:
                print('error in response')
                time.sleep(60)
                response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            reply = response.choices[0].message.content
            reply = re.sub('\s+', '', reply).lower()
            print("*"*50)
            print('reply:', reply)
            reply = reply.replace('確率', '信頼度')

            if args.task_name == 'cls':
                try:
                    matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)
                    if '信頼度' not in reply:
                        answers['信頼度'] = ''
                    else:
                        if '信頼度' not in answers:
                            start_index = reply.find('信頼度')
                            answers['信頼度'] = reply[start_index + 3:]
                except:
                    if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                        if '信頼度' in reply and '理由' in reply:
                            start_index = reply.find('信頼度')
                            end_index = reply.find('理由')
                            answers['信頼度'] = reply[start_index + 3:end_index]
                        elif '信頼度' in reply:
                            start_index = reply.find('信頼度')
                            answers['信頼度'] = reply[start_index + 3:]
                        else:
                            answers['信頼度'] = ''
                    if 'はい' in reply and 'いいえ' not in reply:
                        answers['回答'] = 'はい'
                        answers['理由'] = reply
                    elif 'いいえ' in reply and 'はい' not in reply:
                        answers['回答'] = 'いいえ'
                        answers['理由'] = reply
                    else:
                        answers['回答'] = ''
                        answers['理由'] = reply
                answers['teacher_preds'] = convert_to_label(answers['回答'])
                answers['teacher_rationale'] = answers['理由']
                del answers['回答']
                del answers['理由']
                if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                    answers['confidence'] = answers['信頼度']
                    del answers['信頼度']
                    answers['confidence'] = convert_confidence(answers['confidence'])
                print("*" * 50)
                print('answers:', answers)
                return answers
            else:
                label_index = reply.find('ラベル')
                label_range = reply[label_index:]
                pattern = r'\d+'
                label = int(re.findall(pattern, label_range)[0])
                reply = reply.replace('ラベル', '回答')
                try:
                    matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)
                    if '信頼度' not in reply:
                        answers['信頼度'] = ''
                    else:
                        if '信頼度' not in answers:
                            start_index = reply.find('信頼度')
                            answers['信頼度'] = reply[start_index + 3:]
                except:
                    if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                        if '信頼度' in reply and '理由' in reply:
                            start_index = reply.find('信頼度')
                            end_index = reply.find('理由')
                            answers['信頼度'] = reply[start_index + 3:end_index]
                        elif '信頼度' in reply:
                            start_index = reply.find('信頼度')
                            answers['信頼度'] = reply[start_index + 3:]
                        else:
                            answers['信頼度'] = ''
                    answers['回答'] = label
                    answers['理由'] = reply
                answers['teacher_preds'] = answers['回答']
                answers['teacher_rationale'] = answers['理由']
                del answers['回答']
                del answers['理由']
                if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                    answers['confidence'] = answers['信頼度']
                    del answers['信頼度']
                    answers['confidence'] = convert_confidence(answers['confidence'])
                print("*" * 50)
                print('answers:', answers)
                return answers
    elif args.task_name == 'ce':
        if args.model_name == 'gpt3.5turbo':
            if args.prompt_method == 'dualchecker':
                new_instance = find_similar_text(text['summary'], lookup, args.n_shot,args.task_name)
                new_instance = [
                    generate_rationale(new_instance[i][0], new_instance[i][1], model_type='close', mode=args.task_name)
                    for i in
                    range(len(new_instance))]
                if len(new_example) > 0:
                    new_template = [new_example[-1]] + new_instance
                else:
                    new_template = new_instance
            else:
                new_template = new_example
            template, end_template = ce_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
            if args.reprompt:
                if args.prompt_method == 'dualchecker':
                    text_prompt = [{"role": "system", "content": template}] + [
                        {"role": "user", "content": f"{end_template + text['detail'] + ': '}"}]
                    print("*" * 50)
                    print('start reprompt')
                if args.prompt_method == 'dtot':
                    text_prompt = [{"role": "system", "content": template}] + [
                        {"role": "user", "content": f"{end_template + text['detail'] + ': '}"}]
            else:
                text_prompt = [{"role": "system", "content": template}] + [{"role": "user", "content": f"{end_template + text['summary'] + ': '}"}]
            print("*" * 50)
            print('text prompt:', text_prompt)

            answers = {}
            try:
                response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            except:
                print('error in response')
                time.sleep(60)
                response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
            reply = response.choices[0].message.content
            reply = re.sub('\s+', '', reply).lower()
            print("*"*50)
            print('reply:', reply)
            reply = reply.replace('確率', '信頼度')
            try:
                matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
                answers = json.loads(matches)
                if '信頼度' not in reply:
                    answers['信頼度'] = ''
                else:
                    if '信頼度' not in answers:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
                if '環境効果' not in answers:
                    if '環境効果' in reply:
                        start_index = reply.find('環境効果')
                        end_index = reply[start_index:].find('。')
                        answers['環境効果'] = reply[start_index + 4:end_index]
                    else:
                        answers['環境効果'] = ''
                if '技術' not in answers:
                    if '技術' in reply:
                        start_index = reply.find('技術')
                        end_index = reply[start_index:].find('。')
                        answers['技術'] = reply[start_index + 2:end_index]
                    else:
                        answers['技術'] = ''
            except:
                if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                    if '信頼度' in reply and '理由' in reply:
                        start_index = reply.find('信頼度')
                        end_index = reply.find('理由')
                        answers['信頼度'] = reply[start_index + 3:end_index]
                    elif '信頼度' in reply:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
                    else:
                        answers['信頼度'] = ''
                if '環境効果' in reply:
                    start_index = reply.find('環境効果')
                    end_index = reply[start_index:].find('。')
                    answers['環境効果'] = reply[start_index + 4:end_index]
                else:
                    answers['環境効果'] = ''

                if '技術' in reply:
                    start_index = reply.find('技術')
                    end_index = reply[start_index:].find('。')
                    answers['技術'] = reply[start_index + 2:end_index]
                else:
                    answers['技術'] = ''
                if '理由' in reply:
                    start_index = reply.find('理由')
                    answers['理由'] = reply[start_index + 2:]
                else:
                    answers['理由'] = reply
            answers['teacher_preds'] = {'cause': answers['技術'], 'effect': answers['環境効果']}
            answers['teacher_rationale'] = answers['理由']
            del answers['技術']
            del answers['環境効果']
            del answers['理由']
            if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                answers['confidence'] = answers['信頼度']
                del answers['信頼度']
                answers['confidence'] = convert_confidence(answers['confidence'])
            print("*" * 50)
            print('answers:', answers)
            return answers

def get_substring_from_keyword(input_string, keyword):
    keyword_pos = input_string.find(keyword)
    if keyword_pos == -1:
        return ""
    return input_string[keyword_pos:]

def get_llama(text,model, tokenizer,args,new_example,lookup,simimodel=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    model.eval()

    if args.task_name == 'cls' or args.task_name == 'path':
        if args.prompt_method == 'dualchecker':
            new_instance = find_similar_text(text['summary'], lookup, args.n_shot,args.task_name)
            new_instance = [
                generate_rationale(new_instance[i][0], new_instance[i][1], model_type='open', model=model,tokenizer=tokenizer,mode=args.task_name)
                for i in range(len(new_instance))]
            if len(new_example) > 0:
                new_template = [new_example[-1]] + new_instance
            else:
                new_template = new_instance
        else:
            new_template = new_example
        if args.task_name == 'cls':
            template, end_template = cls_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
        elif args.task_name == 'path':
            template, end_template = path_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
        if args.reprompt:
            if args.prompt_method == 'dualchecker':
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{template}{E_SYS}",
                    prompt=end_template + text['detail'] + ': ',
                    e_inst=E_INST,
                )
                print("*" * 50)
                print('start reprompt')
            if args.prompt_method == 'dtot':
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{template}{E_SYS}",
                    prompt=end_template + text['detail'] + ': ',
                    e_inst=E_INST,
                )
        else:
            prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{template}{E_SYS}",
                prompt=end_template + text['summary'] + ': ',
                e_inst=E_INST,
            )

        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=len(prompt)+100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
        reply = get_substring_from_keyword(output, E_INST)
        reply = re.sub('\s+','',reply).lower()
        example_str = reply.find('例えば')
        reply = reply[:example_str]
        reply = reply.replace('確率', '信頼度')
        answers = {}

        if args.task_name == 'cls':
            try:
                matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
                answers = json.loads(matches)
                if '信頼度' not in reply:
                    answers['信頼度'] = ''
                else:
                    if '信頼度' not in answers:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
            except:
                if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                    if '信頼度' in reply and '理由' in reply:
                        start_index = reply.find('信頼度')
                        end_index = reply.find('理由')
                        answers['信頼度'] = reply[start_index + 3:end_index]
                    elif '信頼度' in reply:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
                    else:
                        answers['信頼度'] = ''

                answer = check_posneg(reply)
                if answer == 0:
                    answers['回答'] = 'いいえ'
                    answers['理由'] = reply
                elif answer == 1:
                    answers['回答'] = 'はい'
                    answers['理由'] = reply
                else:
                    answers['回答'] = ''
                    answers['理由'] = reply
            answers['teacher_preds'] = convert_to_label(answers['回答'])
            answers['teacher_rationale'] = answers['理由']
            del answers['回答']
            del answers['理由']
            if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                answers['confidence'] = answers['信頼度']
                del answers['信頼度']
                answers['confidence'] = convert_confidence(answers['confidence'])
            print("*" * 50)
            print('answers:', answers)
            return answers
        else:
            label_index = reply.find('ラベル')
            label_range = reply[label_index:]
            pattern = r'\d+'
            try:
                label = int(re.findall(pattern, label_range)[0])
            except:
                label = ''
            reply = reply.replace('ラベル', '回答')
            try:
                matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
                answers = json.loads(matches)
                if '信頼度' not in reply:
                    answers['信頼度'] = ''
                else:
                    if '信頼度' not in answers:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
            except:
                if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                    if '信頼度' in reply and '理由' in reply:
                        start_index = reply.find('信頼度')
                        end_index = reply.find('理由')
                        answers['信頼度'] = reply[start_index + 3:end_index]
                    elif '信頼度' in reply:
                        start_index = reply.find('信頼度')
                        answers['信頼度'] = reply[start_index + 3:]
                    else:
                        answers['信頼度'] = ''
                answers['回答'] = label
                answers['理由'] = reply
            answers['teacher_preds'] = answers['回答']
            answers['teacher_rationale'] = answers['理由']
            del answers['回答']
            del answers['理由']
            if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                answers['confidence'] = answers['信頼度']
                del answers['信頼度']
                answers['confidence'] = convert_confidence(answers['confidence'])
            print("*" * 50)
            print('answers:', answers)
            return answers
    elif args.task_name == 'ce':
        if args.prompt_method == 'dualchecker':
            new_instance = find_similar_text(text['summary'], lookup, args.n_shot,args.task_name)
            new_instance = [
                generate_rationale(new_instance[i][0], new_instance[i][1], model_type='open', model=model,tokenizer=tokenizer,mode=args.task_name)
                for i in range(len(new_instance))]
            if len(new_example) > 0:
                new_template = [new_example[-1]] + new_instance
            else:
                new_template = new_instance
        else:
            new_template = new_example
        template, end_template = ce_template(args.prompt_method, args.n_shot, new_template, args.reprompt)
        if args.reprompt:
            if args.prompt_method == 'dualchecker':
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{template}{E_SYS}",
                    prompt=end_template + text['detail'] + ': ',
                    e_inst=E_INST,
                )
                print("*" * 50)
                print('start reprompt')
            if args.prompt_method == 'dtot':
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{template}{E_SYS}",
                    prompt=end_template + text['detail'] + ': ',
                    e_inst=E_INST,
                )
        else:
            prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{template}{E_SYS}",
                prompt=end_template + text['summary'] + ': ',
                e_inst=E_INST,
            )

        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=len(prompt)+100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
        reply = get_substring_from_keyword(output, E_INST)
        reply = re.sub('\s+','',reply).lower()
        reply = reply.replace('確率', '信頼度')
        answers = {}
        print("*" * 50)
        print('prompt:', prompt)
        print("*" * 50)
        print('reply:', reply)
        try:
            matches = re.findall(r"\{.*?\}", reply, re.DOTALL)[0].replace("'", '"')
            answers = json.loads(matches)
            if '信頼度' not in reply:
                answers['信頼度'] = ''
            else:
                if '信頼度' not in answers:
                    start_index = reply.find('信頼度')
                    answers['信頼度'] = reply[start_index + 3:]
            if '環境効果' not in answers or len(answers['環境効果']) < 5 :
                if '環境効果:' in reply:
                    answers['環境効果'] = find_longest_substring_between_keywords(reply, '環境効果:', '信頼度')
                elif '環境効果は' in reply:
                    answers['環境効果'] = find_longest_substring_between_keywords(reply, '環境効果は', '。')
                else:
                    answers['環境効果'] = ''
            if '技術' not in answers or len(answers['技術']) < 5:
                if '技術:' in reply:
                    answers['技術'] = find_longest_substring_between_keywords(reply, '技術:', '環境効果')
                elif '技術は' in reply:
                    answers['技術'] = find_longest_substring_between_keywords(reply, '技術は', '。')
                else:
                    answers['技術'] = ''
            if '理由' not in answers:
                if '理由' in reply:
                    start_index = reply.find('理由')
                    answers['理由'] = reply[start_index + 2:]
                else:
                    answers['理由'] = reply
            if answers['技術'] == '' or answers['環境効果'] == '':
                raise Exception
        except:
            if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
                if '信頼度' in reply and '理由' in reply:
                    answers['信頼度'] = find_longest_substring_between_keywords(reply, '信頼度:', '理由')
                elif '信頼度' in reply:
                    start_index = reply.find('信頼度')
                    answers['信頼度'] = reply[start_index + 3:]
                else:
                    answers['信頼度'] = ''
            if '環境効果:' in reply:
                answers['環境効果'] = find_longest_substring_between_keywords(reply, '環境効果:', '信頼度')
            elif '環境効果は' in reply:
                answers['環境効果'] = find_longest_substring_between_keywords(reply, '環境効果は', '。')
            else:
                answers['環境効果'] = ''

            if '技術:' in reply:
                answers['技術'] = find_longest_substring_between_keywords(reply, '技術:', '環境効果')
            elif '技術は' in reply:
                answers['技術'] = find_longest_substring_between_keywords(reply, '技術は', '。')
            else:
                answers['技術'] = ''
            if '理由' in reply:
                start_index = reply.find('理由')
                answers['理由'] = reply[start_index + 2:]
            else:
                answers['理由'] = reply
        answers['teacher_preds'] = {'cause': answers['技術'], 'effect': answers['環境効果']}
        answers['teacher_rationale'] = answers['理由']
        del answers['技術']
        del answers['環境効果']
        del answers['理由']
        if args.prompt_method == 'dtot' or args.prompt_method == 'dualchecker':
            answers['confidence'] = answers['信頼度']
            del answers['信頼度']
            answers['confidence'] = convert_confidence(answers['confidence'])
        print("*" * 50)
        print('answers:', answers)
        return answers

class teacher_model():
    def __init__(self, model_type='close',mode=None):
        self.model_type = model_type
        self.instructions = []
        self.which_label = 1
        self.lookup = {}
        self.mode = mode

    def init_llama(self,args):
        model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"
        if args.use_gpu:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map = "auto",low_cpu_mem_usage=True,use_cache=True)
        else:
            device = torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def init_simimodel(self,args):
        self.simimodel = SentenceTransformer(args.simimodel_path)

    def generate_data_embeddings(self, data):
        data = {str(d['summary']): d['label'] for d in data}
        texts = list(data.keys())
        labels = list(data.values())
        embeddings = self.simimodel.encode(texts, convert_to_tensor=True)
        self.lookup = {i: (embeddings[i],texts[i],labels[i]) for i in range(len(texts))}
        print("*************************Embeddings generated*************************")

    def get_results(self, text,args):
        if self.model_type == 'close':
            results = get_gpt(text,args,self.instructions,self.lookup,self.simimodel)
        else:
            results = get_llama(text,self.model, self.tokenizer,args,self.instructions,self.lookup,self.simimodel)
        return results

    def evokd_feedback(self, wrong_texts, true_labels,wrong_preds):
        if self.mode != 'ce':
            if len(wrong_texts) > 4:
                wrong_texts = wrong_texts[:4]
                true_labels = true_labels[:4]
                wrong_preds = wrong_preds[:4]
            if self.mode == 'cls':
                true_labels = ['はい' if label == 1 else 'いいえ' for label in true_labels]
                wrong_preds = ['はい' if pred == 1 else 'いいえ' for pred in wrong_preds]
            batch_texts = [f"'文: {text}','ラベル: {label}', '予測: {pred}'" for text, label, pred in
                           zip(wrong_texts, true_labels, wrong_preds)]
            first_sys = evokd_feedback_tp('start_template', self.mode) + ''.join(batch_texts)
            first_usr = evokd_feedback_tp('middle_template', self.mode)
            if self.model_type == 'close':
                text_prompt = [{"role": "system", "content": first_sys}] + [{"role": "user", "content": first_usr}]
                try:
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                except:
                    time.sleep(60)
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                reply1 = response1.choices[0].message.content
                reply1 = re.sub('\s+', '', reply1).lower()
                snd_sys = reply1
                snd_usr = evokd_feedback_tp('end_template', self.mode)
                text_prompt = [{"role": "system", "content": snd_sys}] + [{"role": "user", "content": snd_usr}]
                try:
                    response2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                except:
                    time.sleep(60)
                    response2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                reply2 = response2.choices[0].message.content
                reply2 = re.sub('\s+', '', reply2).lower()
                try:
                    matches = re.findall(r"\{.*?\}", reply2, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)
                except:
                    answers = {}

            else:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
                self.model.eval()
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=self.tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{first_sys}{E_SYS}",
                    prompt=first_usr,
                    e_inst=E_INST,
                )
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

                with torch.no_grad():
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        max_new_tokens=len(prompt) + 100,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
                reply1 = get_substring_from_keyword(output, E_INST)
                reply1 = re.sub('\s+', '', reply1).lower()
                example_str = reply1.find('例えば')
                reply1 = reply1[:example_str]
                snd_sys = reply1
                snd_usr = evokd_feedback_tp('end_template', self.mode)
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=self.tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{snd_sys}{E_SYS}",
                    prompt=snd_usr,
                    e_inst=E_INST,
                )
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                with torch.no_grad():
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        max_new_tokens=len(prompt) + 100,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
                reply2 = get_substring_from_keyword(output, E_INST)
                reply2 = re.sub('\s+', '', reply2).lower()
                example_str = reply2.find('例えば')
                reply2 = reply2[:example_str]
                try:
                    matches = re.findall(r"\{.*?\}", reply2, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)

                except:
                    answers = {}

            texts = []
            labels = []
            text1 = {}
            text2 = {}
            if self.mode == 'cls':
                for key, value in answers.items():
                    if 'はい' in value and '回答' not in key:
                        texts.append(key)
                        labels.append(1)
                    elif 'いいえ' in value and '回答' not in key:
                        texts.append(key)
                        labels.append(0)
                    elif '文1' in key:
                        text1[value] = False
                    elif '文2' in key:
                        text2[value] = False
                    elif '回答1' in key:
                        text1[list(text1.keys())[0]] = convert_to_label(value)
                    elif '回答2' in key:
                        text2[list(text2.keys())[0]] = convert_to_label(value)

                if text1 and text2:
                    if text1[list(text1.keys())[0]] == 1:
                        texts.append(list(text1.keys())[0])
                        labels.append(1)
                    else:
                        texts.append(list(text2.keys())[0])
                        labels.append(0)
            elif self.mode == 'path':
                for key, value in answers.items():
                    texts.append(key)
                    labels.append(value)
        else:
            if len(wrong_texts) > 4:
                wrong_texts = wrong_texts[:4]
                true_labels = true_labels[:4]
                wrong_preds = wrong_preds[:4]
            batch_texts = [f"'文: {text}','技術: {label['cause']}', '環境効果: {label['effect']}', '予測の技術: {pred['cause']}', '予測の環境効果: {pred['effect']}'" for text, label, pred in
                           zip(wrong_texts, true_labels, wrong_preds)]
            first_sys = evokd_feedback_tp('start_template', self.mode) + ''.join(batch_texts)
            first_usr = evokd_feedback_tp('middle_template', self.mode)
            if self.model_type == 'close':
                text_prompt = [{"role": "system", "content": first_sys}] + [{"role": "user", "content": first_usr}]
                try:
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                except:
                    time.sleep(60)
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                reply1 = response1.choices[0].message.content
                reply1 = re.sub('\s+', '', reply1).lower()
                snd_sys = reply1
                snd_usr = evokd_feedback_tp('end_template', self.mode)
                text_prompt = [{"role": "system", "content": snd_sys}] + [{"role": "user", "content": snd_usr}]
                try:
                    response2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                except:
                    time.sleep(60)
                    response2 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                reply2 = response2.choices[0].message.content
                reply2 = re.sub('\s+', '', reply2).lower()
                try:
                    matches = re.findall(r"\{.*?\}", reply2, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)
                except:
                    answers = reply2

            else:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
                self.model.eval()
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=self.tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{first_sys}{E_SYS}",
                    prompt=first_usr,
                    e_inst=E_INST,
                )
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

                with torch.no_grad():
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        max_new_tokens=len(prompt) + 100,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
                reply1 = get_substring_from_keyword(output, E_INST)
                reply1 = re.sub('\s+', '', reply1).lower()
                example_str = reply1.find('例えば')
                reply1 = reply1[:example_str]
                snd_sys = reply1
                snd_usr = evokd_feedback_tp('end_template', self.mode)
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=self.tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{snd_sys}{E_SYS}",
                    prompt=snd_usr,
                    e_inst=E_INST,
                )
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                with torch.no_grad():
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        max_new_tokens=len(prompt) + 100,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
                reply2 = get_substring_from_keyword(output, E_INST)
                reply2 = re.sub('\s+', '', reply2).lower()
                example_str = reply2.find('例えば')
                reply2 = reply2[:example_str]
                try:
                    matches = re.findall(r"\{.*?\}", reply2, re.DOTALL)[0].replace("'", '"')
                    answers = json.loads(matches)

                except:
                    answers = reply2

            texts = []
            labels = []

            if type(answers) == dict:
                for key, value in answers.items():
                    if type(value) == dict:
                        if '技術1' in value and '環境効果1' in value:
                            texts.append(key)
                            labels.append({'cause': value['技術1'], 'effect': value['環境効果1']})
                        if '技術2' in value and '環境効果2' in value:
                            texts.append(key)
                            labels.append({'cause': value['技術2'], 'effect': value['環境効果2']})
        return texts, labels


    def dualchecker_feedback(self, wrong_text, true_label, instructions=[]):
        self.instructions.extend(instructions)

        if self.mode == 'cls':
            true_label = 'はい' if true_label == 1 else 'いいえ'
            end = wrong_text[:120].rfind('。')
            wrong_text = wrong_text[:end + 1]
            batch_text = f"'文: {wrong_text}','ラベル: {true_label}'。'"
        elif self.mode == 'path':
            path_begin = '本発明から環境課題の解決までの具体的な影響パスの最初の３つのノードは'
            start_index = wrong_text.find(path_begin)
            path = wrong_text[start_index:]
            wrong_text = wrong_text[:start_index][:120]
            end = wrong_text.rfind('。')
            wrong_text = wrong_text[:end + 1] + path
            batch_text = f"'文: {wrong_text}','ラベル: {true_label}'。'"
        elif self.mode == 'ce':
            batch_text = f"'文: {wrong_text}'"

        if self.mode != 'ce':
            first_sys = dualchecker_feedback_tp('start_template', true_label, self.mode) + batch_text
            first_usr = dualchecker_feedback_tp('end_template', true_label, self.mode)
            if self.model_type == 'close':
                text_prompt = [{"role": "system", "content": first_sys}] + [{"role": "user", "content": first_usr}]
                try:
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                except:
                    time.sleep(60)
                    response1 = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=text_prompt)
                reply = response1.choices[0].message.content
                reply = re.sub('\s+', '', reply).lower()
            else:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
                self.model.eval()
                prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                    bos_token=self.tokenizer.bos_token,
                    b_inst=B_INST,
                    system=f"{B_SYS}{first_sys}{E_SYS}",
                    prompt=first_usr,
                    e_inst=E_INST,
                )
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

                with torch.no_grad():
                    output_ids = self.model.generate(
                        token_ids.to(self.model.device),
                        max_new_tokens=len(prompt) + 100,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                output = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
                reply = get_substring_from_keyword(output, E_INST)
                reply = re.sub('\s+', '', reply).lower()
                example_str = reply.find('例えば')
                reply = reply[:example_str]

        if self.mode != 'ce':
            if reply:
                new_instruction = f'例えば、『{wrong_text}』回答：{true_label}。理由：{reply}'
            else:
                new_instruction = ''
        else:
            cause = true_label['cause']
            effect = true_label['effect']
            new_instruction = f'例えば、『{wrong_text}』回答：技術は「{cause}」で、環境効果は「{effect}」です。'
        return new_instruction