
from tqdm import trange
from script.dataloader import load_cls_train_data
from script.get_teacher import teacher_model
import json
import os
from script.get_student import *


def save_json(data,save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w',  encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def run(args, log):

    log.logger.info(f'run_{args.task_name}')
    all_data = load_cls_train_data(dt=f'data/{args.task_name}_train.json')
    if args.part != 'all':
        save_dir = args.part
    else:
        save_dir = ''

    args.path_class = {0: 'エネルギー効率と消費の削減 - 全てのエネルギー消費の削減、効率の向上に関する内容。',
                       1: '再生可能エネルギーと排出ガス削減 - 再生可能エネルギーの利用促進と排出ガス・温室効果ガスの削減に関する内容。',
                       2: '廃棄物管理とリサイクル - 廃棄物の削減、リサイクルの効率化、資源の循環利用に関する内容。',
                       3: '製品開発と技術革新 - 新技術の開発、製品の耐久性と安全性の向上に関する内容。',
                       'エネルギー効率と消費の削減 - 全てのエネルギー消費の削減、効率の向上に関する内容。': 0,
                       '再生可能エネルギーと排出ガス削減 - 再生可能エネルギーの利用促進と排出ガス・温室効果ガスの削減に関する内容。': 1,
                       '廃棄物管理とリサイクル - 廃棄物の削減、リサイクルの効率化、資源の循環利用に関する内容。': 2,
                       '製品開発と技術革新 - 新技術の開発、製品の耐久性と安全性の向上に関する内容。': 3}

    for i in range(len(all_data)):
        if args.task_name == 'path':
            all_data[i]['summary'] = all_data[i]['summary'] + '。' + all_data[i]['path']
            if args.model_name == 'llama2':
                stop_index = all_data[i]['detail'][:500].rfind('。')
                all_data[i]['detail'] = all_data[i]['detail'][:stop_index] + '。' + all_data[i]['path']
            else:
                all_data[i]['detail'] = all_data[i]['detail'] + '。' + all_data[i]['path']
            all_data[i]['label'] = args.path_class[all_data[i]['label']]
        elif args.task_name == 'ce':
            if args.model_name == 'llama2':
                stop_index = all_data[i]['detail'][:100].rfind('。')
                all_data[i]['detail'] = all_data[i]['summary'] + all_data[i]['detail'][:stop_index]
            else:
                all_data[i]['detail'] = all_data[i]['summary'] + all_data[i]['detail']
            all_data[i]['label'] = {'cause': all_data[i]['cause'], 'effect': all_data[i]['effect']}

    if os.path.exists(f'teacher_predictions/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json'):
        data = load_cls_train_data(f'teacher_predictions/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')
    else:
        data = all_data[:int(len(all_data) * args.train_ratio)]

    teacher = teacher_model(model_type=args.model_type,mode=args.task_name)
    teacher.init_simimodel(args)

    if args.task_name == 'path':
        num_labels = 4
    elif args.task_name == 'cls':
        num_labels = 2
    elif args.task_name == 'ce':
        num_labels = 'token_classification'

    if args.prompt_method == 'evokd':
        args.student_threshold = 0.95

    if args.prompt_method == 'evokd' or args.prompt_method == 'dualchecker':
        args.save_path = f'save_model/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}' + save_dir
        trainer = TextClassifierTrainer(args.student_path,num_labels)
        trainer.set_parameters(save_path=args.save_path,prompt_method=args.prompt_method,teacher_model=teacher,batch_size=args.batch_size,lr=args.lr,epoch=args.epoch,max_length=args.max_length,student_threshold=args.student_threshold)
        if args.prompt_method == 'dualchecker':
            teacher.generate_data_embeddings(all_data[:int(len(all_data) * args.train_ratio)])

    if args.model_type == 'open':
        teacher.init_llama(args)


    for i in trange(len(data), desc='get teacher predictions'):
        if 'teacher_preds' in data[i].keys():
            if data[i]['teacher_preds'] != '':
                print(f'{i} already has teacher predictions')
                continue
        try:
            teacher_predictions = teacher.get_results(data[i], args)
        except Exception as e:
            raise ValueError(f'error in line 66 of run.py get_results: {e}')

        if args.prompt_method == 'dtot':
            data[i]['teacher_confidence'] = teacher_predictions['confidence']
            if args.prompt_method == 'dtot' and args.model_name == 'llama2':
                args.teacher_threshold = 0.8
            elif args.prompt_method == 'dtot' and args.model_name == 'gpt3.5turbo':
                args.teacher_threshold = 0.9

            if teacher_predictions['confidence'] == '' or teacher_predictions['confidence'] < args.teacher_threshold:
                data[i]['teacher_confidence_before'] = teacher_predictions['confidence']
                data[i]['teacher_preds_before'] = teacher_predictions['teacher_preds']
                data[i]['teacher_rationale_before'] = teacher_predictions['teacher_rationale']
                args.reprompt = True
                confidence = teacher_predictions['confidence']
                print('*' * 50)
                print(f'confidence {confidence} is low, reprompt ...')
                try:
                    teacher_predictions = teacher.get_results(data[i], args)
                    data[i]['teacher_confidence'] = teacher_predictions['confidence']
                except Exception as e:
                    raise ValueError(f'error in line 66 of run.py get_results: {e}')
                args.reprompt = False

        elif args.prompt_method == 'dualchecker':
            data[i]['teacher_confidence'] = teacher_predictions['confidence']
            if args.prompt_method == 'dualchecker' and args.model_name == 'llama2':
                args.teacher_threshold = 0.75
            elif args.prompt_method == 'dualchecker' and args.model_name == 'gpt3.5turbo':
                args.teacher_threshold = 0.85

            if args.part == 'teacher' or args.part == 'all':
                if teacher_predictions['confidence'] == '' or teacher_predictions['confidence'] < args.teacher_threshold:
                    data[i]['teacher_confidence_before'] = teacher_predictions['confidence']
                    data[i]['teacher_preds_before'] = teacher_predictions['teacher_preds']
                    data[i]['teacher_rationale_before'] = teacher_predictions['teacher_rationale']
                    args.reprompt = True
                    confidence = teacher_predictions['confidence']
                    print('*' * 50)
                    print(f'confidence {confidence} is low, reprompt ...')
                    try:
                        teacher_predictions = teacher.get_results(data[i], args)
                        data[i]['teacher_confidence'] = teacher_predictions['confidence']
                    except Exception as e:
                        raise ValueError(f'error in line 66 of run.py get_results: {e}')
                    args.reprompt = False

        if args.prompt_method == 'evokd':
            summary = data[i]['summary']
            if args.task_name == 'path':
                path_begin = '本発明から環境課題の解決までの具体的な影響パスの最初の３つのノードは'
                start_index = summary.find(path_begin)
                path = summary[start_index:]
                summary = summary[:start_index][:450]
                end =summary.rfind('。')
                summary = summary[:end+1] + path
            try:
                trainer.add_training_data(summary, teacher_predictions['teacher_preds'],data[i]['label'])
            except Exception as e:
                print(f'error in line 92 of run.py add_training_data: {e}')
                continue

        if args.prompt_method == 'dualchecker':
            if args.part == 'student' or args.part == 'all':
                summary = data[i]['summary']
                if args.task_name == 'path':
                    path_begin = '本発明から環境課題の解決までの具体的な影響パスの最初の３つのノードは'
                    start_index = summary.find(path_begin)
                    path = summary[start_index:]
                    summary = summary[:start_index][:450]
                    end = summary.rfind('。')
                    summary = summary[:end + 1] + path
                try:
                    trainer.add_training_data(summary, teacher_predictions['teacher_preds'], data[i]['label'])
                except Exception as e:
                    print(f'error in line 92 of run.py add_training_data: {e}')
                    continue

        print("*" * 100)

        data[i]['teacher_preds'] = teacher_predictions['teacher_preds']
        data[i]['teacher_rationale'] = teacher_predictions['teacher_rationale']
        save_json(data, f'teacher_predictions/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')
        print(f'save teacher predictions to teacher_predictions/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')

    if args.prompt_method == 'evokd':
        trainer.save_model()
        new_data = [{'texts':trainer.new_texts[i],'labels':trainer.new_labels[i]} for i in range(len(trainer.new_texts))]
        if new_data != []:
            save_json(new_data, f'new_data/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')

            print(f'save new data to new_data/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')

    elif args.prompt_method == 'dualchecker':
        trainer.save_model()
        new_fewshot_examples = [{f'{i}':(trainer.instructions[i],trainer.instructions_probs[i]) for i in range(len(trainer.instructions))}]
        if trainer.instructions != []:
            save_json(new_fewshot_examples, f'new_fewshot_examples/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')
            print(f'save new fewshot examples to new_fewshot_examples/{args.task_name}/{args.prompt_method}/{args.model_name}_{args.n_shot}{save_dir}.json')
