from dataloader import load_cls_train_data
from transformers import BertJapaneseTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaForTokenClassification
import argparse
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
import numpy as np
import copy
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizerFast
from transformers.models.bert_japanese.tokenization_bert_japanese import MecabTokenizer

class MecabPreTokenizer(MecabTokenizer):
  def mecab_split(self,i,normalized_string):
    t=str(normalized_string)
    e=0
    z=[]
    for c in self.tokenize(t):
      s=t.find(c,e)
      e=e if s<0 else s+len(c)
      z.append((0,0) if s<0 else (s,e))
    return [normalized_string[s:e] for s,e in z if e>0]
  def pre_tokenize(self,pretok):
    pretok.split(self.mecab_split)

class BertMecabTokenizerFast(BertTokenizerFast):
  def __init__(self,vocab_file,do_lower_case=False,tokenize_chinese_chars=False,**kwargs):
    from tokenizers.pre_tokenizers import PreTokenizer,BertPreTokenizer,Sequence
    super().__init__(vocab_file=vocab_file,do_lower_case=do_lower_case,tokenize_chinese_chars=tokenize_chinese_chars,**kwargs)
    d=kwargs["mecab_kwargs"] if "mecab_kwargs" in kwargs else {"mecab_dic":"ipadic"}
    self._tokenizer.pre_tokenizer=Sequence([PreTokenizer.custom(MecabPreTokenizer(**d)),BertPreTokenizer()])

class CustomCollate:
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        input_ids_padded = pad_sequence([torch.tensor(seq) for seq in input_ids],
                                        batch_first=True, padding_value=tokenizer.pad_token_id)

        labels_padded = pad_sequence([torch.tensor(seq) for seq in labels],
                                     batch_first=True, padding_value=-100)

        return {'input_ids': input_ids_padded, 'labels': labels_padded}

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def from_token_to_char(tkn, valid_text, offset_mapping, text, labeled_text, mode='cuz'):
    try:
        start_index = text.find(tkn[0])
    except:
        start_index = -1
    try:
        end_index = text.rfind(tkn[-1])
    except:
        end_index = -1
    if start_index > end_index:
        start_index = end_index

    found = False
    for i in range(len(offset_mapping)):
        if  offset_mapping[i][0] <= start_index and offset_mapping[i][1] >= start_index:
            start_index = i
            found = True
            break
    if not found:
        start_index = -1
    found = False
    for i in range(len(offset_mapping)):
        if  offset_mapping[i][0] <= end_index and offset_mapping[i][1] >= end_index:
            end_index = i
            found = True
            break
    if not found:
        end_index = -1


    if mode == 'cuz' and start_index != -1 and end_index != -1:
        labeled_text[start_index:end_index + 1] = [1] + [2] * (end_index - start_index)
    elif mode == 'efc' and start_index != -1 and end_index != -1:
        labeled_text[start_index:end_index + 1] = [3] + [4] * (end_index - start_index)

    return labeled_text


def label_text(text, cause, effect, tokenizer):
    valid_text = tokenizer(text)
    offset_mapping = tokenizer(text, return_offsets_mapping=True)['offset_mapping']

    if len(valid_text['input_ids']) != len(offset_mapping):
        raise ValueError(len(valid_text['input_ids']), len(offset_mapping))

    labeled_text = np.zeros(len(valid_text['input_ids']), dtype=int)
    labeled_text = from_token_to_char(cause, valid_text['input_ids'], offset_mapping, text, labeled_text, mode='cuz')
    labeled_text = from_token_to_char(effect, valid_text['input_ids'], offset_mapping, text, labeled_text, mode='efc')

    return labeled_text.tolist()

def genrate_entity_label(inputs, answers,cuz,efc):
    real_labels = []
    startcuz = False
    startefc = False
    for i in range(len(inputs)):
        pairs = {'cause': '', 'effect': ''}
        if answers[i] in cuz and startcuz == False:
            pairs['cause'] += inputs[i]
            startcuz = True
        elif answers[i] in cuz and startcuz == True:
            pairs['cause'] += inputs[i]
        elif answers[i] not in cuz and startcuz == True:
            startcuz = False
        if answers[i] in efc and startefc == False:
            pairs['effect'] += inputs[i]
            startefc = True
        elif answers[i] in efc and startefc == True:
            pairs['effect'] += inputs[i]
        elif answers[i] not in efc and startefc == True:
            startefc = False
        real_labels.append(pairs)
    return real_labels

def calculate_ner(answers,logits,inputs,mode='soft'):
    cuz = [1,2]
    efc = [3,4]

    real_labels = genrate_entity_label(inputs, answers,cuz,efc)
    predict_labels = genrate_entity_label(inputs, logits,cuz,efc)

    cnt = 0
    tp, fp, fn = 0, 0, 0

    for i in range(len(real_labels)):
        ans_cause = real_labels[i].get('cause', '')
        ans_effect = real_labels[i].get('effect', '')


        pred_cause = predict_labels[i].get('cause', '')
        pred_effect = predict_labels[i].get('effect', '')

        print(ans_cause,ans_effect)
        print(pred_cause,pred_effect)

        true_tokens = set(list(ans_cause + ans_effect))
        pred_tokens = set(list(pred_cause + pred_effect))

        tp += len(true_tokens & pred_tokens)
        fp += len(pred_tokens - true_tokens)
        fn += len(true_tokens - pred_tokens)

    p_soft = tp / (tp + fp) if tp + fp > 0 else 0
    r_soft = tp / (tp + fn) if tp + fn > 0 else 0
    s_f1 = 2 * p_soft * r_soft / (p_soft + r_soft) if p_soft + r_soft > 0 else 0

    return p_soft, r_soft, s_f1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--teacher', type=str, default='synthetic')
    parser.add_argument('--training_content', type=str, default='label')
    parser.add_argument('--ft_model', type=str, default='roberta')
    parser.add_argument('--task', type=str, default='ce')
    parser.add_argument('--method', type=str, default='dualchecker')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--n_shot', type=int, default=5)
    return parser.parse_args()

args = parse_args()
if args.teacher != 'human':
    method = args.method
    n_shot = args.n_shot
    training_content = args.training_content

data_path = f'../data/{args.task}_train.json'


if args.ft_model == 'roberta':
    path = 'rinna/japanese-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tokenizer.do_lower_case = True
    if args.task == 'cls':
        model = RobertaForSequenceClassification.from_pretrained(path)
    elif args.task == 'path':
        model = RobertaForSequenceClassification.from_pretrained(path, num_labels=4)
    else:
        model = RobertaForTokenClassification.from_pretrained(path, num_labels=5)
else:
    path = '../model/patent_roberta'
    if args.task == 'cls':
        tokenizer = BertJapaneseTokenizer.from_pretrained(path, word_tokenizer_type="sudachi",
                                                          subword_tokenizer_type="wordpiece",
                                                          sudachi_kwargs={"sudachi_split_mode": "C"})
        model = RobertaForSequenceClassification.from_pretrained(path)
    elif args.task == 'path':
        tokenizer = BertJapaneseTokenizer.from_pretrained(path, word_tokenizer_type="sudachi",
                                                          subword_tokenizer_type="wordpiece",
                                                          sudachi_kwargs={"sudachi_split_mode": "C"})
        model = RobertaForSequenceClassification.from_pretrained(path, num_labels=4)
    else:
        tokenizer = BertMecabTokenizerFast.from_pretrained(path)
        model = RobertaForTokenClassification.from_pretrained(path, num_labels=5)

data = load_cls_train_data(dt=data_path)
if args.teacher != 'human':
    teacher_path = f'../teacher_predictions/{args.task}/{args.method}/{args.model_name}_{args.n_shot}_shot.json'
    teacher_predictions = load_cls_train_data(teacher_path)
    train_data = data[:int(len(data) * args.train_ratio)]
    for i in range(len(train_data)):
        id = train_data[i]['id']
        train_data[i]['teacher_rationale'] = ''
        if args.task != 'ce':
            for j in range(len(teacher_predictions)):
                if 'teacher_preds' in teacher_predictions[j].keys():
                    if teacher_predictions[j]['id'] == id and teacher_predictions[j]['teacher_preds'] in [0, 1, 2, 3]:
                        train_data[i]['label'] = teacher_predictions[j]['teacher_preds']
                        train_data[i]['teacher_rationale'] = teacher_predictions[j][
                            'teacher_rationale'] if 'teacher_rationale' in teacher_predictions[j].keys() else ''
                        train_data[i]['teacher_preds'] = teacher_predictions[j]['teacher_preds']
                        break

        elif args.task == 'ce':
            for j in range(len(teacher_predictions)):
                if 'teacher_preds' in teacher_predictions[j].keys():
                    if teacher_predictions[j]['id'] == id and type(teacher_predictions[j]['teacher_preds']) == dict:
                        train_data[i]['cause'] = teacher_predictions[j]['teacher_preds']['cause']
                        train_data[i]['effect'] = teacher_predictions[j]['teacher_preds']['effect']
                        train_data[i]['teacher_rationale'] = teacher_predictions[j][
                            'teacher_rationale'] if 'teacher_rationale' in teacher_predictions[j].keys() else ''
                        train_data[i]['teacher_preds'] = teacher_predictions[j]['teacher_preds']
                        break


    train_data = [d for d in train_data if 'teacher_preds' in d.keys()]

    if training_content == 'rationale':
        for i in range(len(train_data)):
            summary = train_data[i]['summary']
            rationale = train_data[i]['teacher_rationale'].replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace('"', '').replace("'", '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
            if '理由' in rationale:
                start_index = rationale.find('理由')
                rationale = rationale[start_index + 2:]
            if rationale != '':
                if args.task == 'cls' and train_data[i]['label'] == 0:
                    start_template = 'このテキストがグリーンイノベーションのカテゴリに属するか?'
                    end_template = ' この理由について考えましょう: '
                    train_data[i]['summary'] = start_template + '「' + summary + '」' + end_template + rationale
                elif args.task == 'path':
                    start_template = 'この文はグリーンイノベーションに関するもので、技術が最終的に解決可能な環境課題を以下のラベル(0,1,2,3)で分類してください: 0:エネルギー効率と消費の削減 - 全てのエネルギー消費の削減、効率の向上に関する内容、1:再生可能エネルギーと排出ガス削減 - 再生可能エネルギーの利用促進と排出ガス・温室効果ガスの削減に関する内容、2:廃棄物管理とリサイクル - 廃棄物の削減、リサイクルの効率化、資源の循環利用に関する内容、3:製品開発と技術革新 - 新技術の開発、製品の耐久性と安全性の向上に関する内容。'
                    end_template = ' この理由について考えましょう: '
                    path_content = train_data[i]['path']
                    train_data[i]['summary'] = start_template + '「' + summary +  path_content + '」' + end_template + rationale
                elif args.task == 'ce':
                    start_template = 'この文はグリーンイノベーションに関する文で、文の中にある技術と最終的に達成される環境効果は何か？'
                    end_template = ' この理由について考えましょう: '
                    train_data[i]['summary'] = start_template + '「' + summary + '」' + end_template + rationale
            else:
                if args.task == 'path':
                    start_template = 'この文はグリーンイノベーションに関するもので、技術が最終的に解決可能な環境課題を以下のラベル(0,1,2,3)で分類してください: 0:エネルギー効率と消費の削減 - 全てのエネルギー消費の削減、効率の向上に関する内容、1:再生可能エネルギーと排出ガス削減 - 再生可能エネルギーの利用促進と排出ガス・温室効果ガスの削減に関する内容、2:廃棄物管理とリサイクル - 廃棄物の削減、リサイクルの効率化、資源の循環利用に関する内容、3:製品開発と技術革新 - 新技術の開発、製品の耐久性と安全性の向上に関する内容。'
                    path_content = train_data[i]['path']
                    train_data[i]['summary'] = start_template + '「' + summary + path_content + '」'


else:
    train_data = data[:int(len(data) * args.train_ratio)]

train_texts = [d['summary'][:512] for d in train_data]

if args.task == 'cls':
    train_labels = [d['label'] for d in train_data]
    val_data = data[int(len(data) * args.train_ratio):]
    val_texts = [d['summary'] for d in val_data]
    val_labels = [d['label'] for d in val_data]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
elif args.task == 'path':
    train_labels = [d['label'] for d in train_data]
    val_data = data[int(len(data) * args.train_ratio):]
    val_texts = [d['summary'] for d in val_data]
    val_labels = [d['label'] for d in val_data]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
elif args.task == 'ce':
    val_data = data[int(len(data) * args.train_ratio):]
    val_texts = [d['summary'] for d in val_data]
    train_labels, val_labels = [], []
    train_texts_ = copy.deepcopy(train_texts)
    train_texts = []
    for d in train_data:
        label = label_text(d['summary'], d['cause'], d['effect'],tokenizer)
        if set(label) == {0} or len(set(label)) < 2:
            continue
        else:
            train_labels.append(label)
            train_texts.append(d['summary'])
    for d in val_data:
        label = label_text(d['summary'], d['cause'], d['effect'],tokenizer)
        val_labels.append(label)
    train_encodings = tokenizer(train_texts)
    val_encodings = tokenizer(val_texts)

print(f"Total data size: {len(data)}")
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

if args.task == 'cls':
    batch_size = 64
else:
    batch_size = 4

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5,
)


if args.task != 'ce':
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)

    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    preds = preds.flatten()
    labels = labels.flatten()

    report = classification_report(labels, preds, digits=4)
    print(report)
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomCollate()
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)

    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    texts = [tokenizer.convert_ids_to_tokens(val_encodings.data['input_ids'][i]) for i in range(len(val_encodings.data['input_ids']))]
    preds = [preds[i].tolist() for i in range(len(preds))]
    labels = [labels[i].tolist() for i in range(len(labels))]
    for i in range(len(texts)):
        length = len(texts[i])
        preds[i] = preds[i][:length]
        labels[i] = labels[i][:length]

    labels_ = [item for sublist in labels for item in sublist]
    preds_ = [item for sublist in preds for item in sublist]
    labels,preds = [],[]
    for i in range(len(preds_)):
        if labels_[i] != 0:
            if labels_[i] == 1:
                labels_[i] = 2
            elif labels_[i] == 3:
                labels_[i] = 4
            if preds_[i] == 1:
                preds_[i] = 2
            elif preds_[i] == 3:
                preds_[i] = 4
            labels.append(labels_[i])
            preds.append(preds_[i])

    report = classification_report(labels, preds, digits=4)
    print(report)



