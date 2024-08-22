
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import BertJapaneseTokenizer, RobertaForSequenceClassification
from transformers import RobertaForTokenClassification

from torch.utils.data import Dataset
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizerFast
from transformers.models.bert_japanese.tokenization_bert_japanese import MecabTokenizer

def from_token_to_char(tkn,valid_text,offset_mapping,text,labeled_text,mode='cuz'):
    start_tkn, end_tkn = 0, 0
    start_str_, end_str_ = 0, 0
    founded = False
    for i in range(len(valid_text)):
        start_str = offset_mapping[i][0]
        end_str = offset_mapping[i][1]
        if text[start_str:end_str] == tkn[:(end_str - start_str)] and founded == False:
            founded = True
            start_str_ = start_str
            end_str_ = end_str
            start_tkn = i
            end_tkn = i
        elif text[start_str_:end_str] not in tkn and founded == True:
            founded = False
            if text[start_str:end_str] == tkn[:(end_str - start_str)]:
                founded = True
                start_str_ = start_str
                end_str_ = end_str
                start_tkn = i
                end_tkn = i
        elif text[start_str_:end_str] != tkn and founded == True:
            end_str_ = end_str
            end_tkn = i
        elif text[start_str_:end_str] == tkn and founded == True:
            end_str_ = end_str
            end_tkn = i
            break
    if text[start_str_:end_str_] == tkn:
        if mode == 'cuz':
            labeled_text[start_tkn:end_tkn + 1] = [1] + [2] * (end_tkn - start_tkn)
        elif mode == 'efc':
            labeled_text[start_tkn:end_tkn + 1] = [3] + [4] * (end_tkn - start_tkn)
    return labeled_text

def label_text(text, cause,effect,tokenizer):
    valid_text = tokenizer(text)
    offset_mapping = tokenizer(text,return_offsets_mapping=True)['offset_mapping']
    if len(valid_text['input_ids']) != len(offset_mapping):
        raise ValueError(len(valid_text['input_ids']),len(offset_mapping))
    labeled_text = np.zeros(len(valid_text['input_ids']), dtype=int)
    labeled_text = from_token_to_char(cause,valid_text['input_ids'],offset_mapping,text,labeled_text,mode='cuz')
    labeled_text = from_token_to_char(effect, valid_text['input_ids'], offset_mapping, text, labeled_text, mode='efc')
    return labeled_text.tolist()

def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None:
                continue
            if entity_pointer[1] != label.split('-')[1]:
                continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark

def return_cause_effect_from_entity(entity_mark_dict, label):
    causal_dict = {}
    for key, value in entity_mark_dict.items():
        causal_dict[key] = ''.join(label[key[0]:key[0] + len(value)])
    pairs = {'cause': '', 'effect': ''}
    max_cuz_len = 0
    max_efc_len = 0
    duplicate = set()
    cause, effect = '', ''

    for key, value in causal_dict.items():
        if key[-1] == 'cuz' and value not in duplicate:
            cause = value
            duplicate.add(value)
        elif key[-1] == 'efc' and value not in duplicate:
            effect = value
            duplicate.add(value)
        else:
            continue
        current_cause_len = len(cause)
        current_effect_len = len(effect)
        if current_cause_len > max_cuz_len:
            pairs['cause'] = cause
            max_cuz_len = current_cause_len
        if current_effect_len > max_efc_len:
            pairs['effect'] = effect
            max_efc_len = current_effect_len
    return pairs

def custom_collate_fn(batch):
    sequences, labels = zip(*batch)

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    labels = torch.stack(labels, dim=0)

    return padded_sequences, labels

def custom_collate_fn_ner(batch):
    input_ids = [torch.tensor(item[0]).clone().detach() for item in batch]
    labels = [torch.tensor(item[1]).clone().detach() for item in batch]

    input_ids_padded = pad_sequence([torch.tensor(seq) for seq in input_ids],
                                    batch_first=True, padding_value=0)

    labels_padded = pad_sequence([torch.tensor(seq) for seq in labels],
                                 batch_first=True, padding_value=-100)

    print('input_ids_padded shape:',input_ids_padded.shape)
    print('labels_padded shape:',labels_padded.shape)
    return input_ids_padded, labels_padded

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

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length',
                                return_tensors='pt')

        return tokens['input_ids'].squeeze().to(self.device), torch.tensor(label).to(self.device)

class ner_TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.texts)

    def pad_labels(self, label, max_len):
        if len(label) > max_len:
            label = label[:max_len]
        else:
            label = label + [-100] * (max_len - len(label))
        return torch.tensor(label).to(self.device)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length',
                                return_tensors='pt')

        input_ids = tokens['input_ids'].squeeze().to(self.device)

        label_padded = self.pad_labels(label, self.max_length)

        return input_ids, label_padded

class TextClassifierTrainer:
    def __init__(self, model_path,num_labels=2):
        self.num_labels = num_labels
        self.model_path = model_path

        if num_labels == 2:
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_path, word_tokenizer_type="sudachi",
                                                              subword_tokenizer_type="wordpiece",
                                                              sudachi_kwargs={"sudachi_split_mode": "C"})
            self.student_model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,device_map = "auto")
        elif num_labels == 4:
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_path, word_tokenizer_type="sudachi",
                                                              subword_tokenizer_type="wordpiece",
                                                              sudachi_kwargs={"sudachi_split_mode": "C"})
            self.student_model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,device_map = "auto")
        else:
            self.tokenizer = BertMecabTokenizerFast.from_pretrained(model_path)
            self.student_model = RobertaForTokenClassification.from_pretrained(model_path, num_labels=5,device_map = "auto")

        self.optimizer = None
        self.scheduler = None
        self.texts = []
        self.labels = []
        self.gold_labels = []
        self.new_texts = []
        self.new_labels = []
        self.instructions = []
        self.instructions_probs = []
        self.which_label = 0
        self.teacher = None
        self.batch_size = None
        self.epochs = None
        self.max_length = None

    def set_parameters(self, save_path,prompt_method, teacher_model, batch_size, lr, epoch, max_length,student_threshold):
        self.save_path = save_path
        self.prompt_method = prompt_method
        self.teacher = teacher_model
        self.batch_size = batch_size
        self.epochs = epoch
        self.max_length = max_length
        self.student_threshold = student_threshold

        self.optimizer = AdamW(self.student_model.parameters(), lr=lr)
        self.total_steps = self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=500,
                                                         num_training_steps=self.total_steps)

    def add_training_data(self, text, label, gold_label):
        self.texts.append(text)
        self.labels.append(label)
        if gold_label is not None:
            self.gold_labels.append(gold_label)
        print('label:',gold_label)
        print('teacher:',label)
        if len(self.labels) == self.batch_size:
            self.labels = [self.labels[i] if type(self.labels[i]) != str else self.gold_labels[i] for i in range(len(self.labels))]
            print("*" * 100)
            print(f"Training on {len(self.labels)} samples")
            self.train_batch()

    def feedback(self, wrong_texts, true_labels, wrong_preds=None):
        if self.prompt_method == 'evokd':
            new_text,new_label = self.teacher.evokd_feedback(wrong_texts, true_labels,wrong_preds)
            return new_text, new_label
        elif self.prompt_method == 'dualchecker':
            new_instruction = self.teacher.dualchecker_feedback(wrong_texts, true_labels,self.instructions)
            return new_instruction

    def train_batch(self):
        self.student_model.train()
        epoch_loss = 0

        if type(self.num_labels) != int:
            self.labels = [label_text(self.texts[i], self.labels[i]['cause'], self.labels[i]['effect'], self.tokenizer)
                           for i in range(len(self.labels))]

            batch_dataset = ner_TextDataset(self.texts, self.labels, self.tokenizer, self.max_length)
            batch_loader = DataLoader(batch_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=custom_collate_fn_ner)
        else:
            batch_dataset = TextDataset(self.texts, self.labels, self.tokenizer, self.max_length)
            batch_loader = DataLoader(batch_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=custom_collate_fn)
        wrong_texts = []
        wrong_preds = []
        true_labels = []
        gold_labels = []
        student_probs = []

        if type(self.num_labels) == int:
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")
                for batch in tqdm(batch_loader):
                    self.optimizer.zero_grad()
                    input_ids, labels = batch

                    outputs = self.student_model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)

                    if self.prompt_method == 'evokd' and epoch == self.epochs - 1:
                        for j in range(len(preds)):
                            if max_probs[j].item() < self.student_threshold:
                                pred = 1 if self.labels[j] == 0 else 0
                                wrong_texts.append(self.texts[j])
                                true_labels.append(self.labels[j])
                                wrong_preds.append(pred)

                    elif self.prompt_method == 'dualchecker' and epoch == self.epochs - 1:
                        hard_index = sorted([j for j, prob in enumerate(max_probs) if preds[j] != self.gold_labels[j]],
                                            key=lambda x: max_probs[x], reverse=False)
                        if hard_index != []:
                            hard_index = [hard_index[0]]
                            wrong_texts.extend([self.texts[j] for j in hard_index])
                            gold_labels.extend([self.gold_labels[j] for j in hard_index])
                            student_probs.extend([max_probs[j].item() for j in hard_index])
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss.item()
        else:
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")
                for batch in tqdm(batch_loader):
                    self.optimizer.zero_grad()
                    input_ids, labels = batch

                    outputs = self.student_model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    probs = torch.softmax(logits, dim=-1)
                    max_probs, preds = torch.max(probs, dim=-1)
                    print('max_probs_shape:',max_probs.shape)
                    print('preds_shape:',preds.shape)

                    if self.prompt_method == 'evokd' and epoch == self.epochs - 1:
                        for j in range(len(preds)):
                            if torch.mean(max_probs[j]).item() < self.student_threshold:
                                wrong_texts.append(self.texts[j])
                                true_labels.append(self.labels[j])
                                wrong_preds.append(preds[j])

                    elif self.prompt_method == 'dualchecker' and epoch == self.epochs - 1:
                        gold_labels_tmp = [label_text(self.texts[i], self.gold_labels[i]['cause'], self.gold_labels[i]['effect'], self.tokenizer)
                           for i in range(len(self.gold_labels))]
                        hard_index = sorted([j for j, prob in enumerate(max_probs) if preds[j] != gold_labels_tmp[j]],key=lambda x: torch.mean(max_probs[x]).item(), reverse=False)
                        if hard_index != []:
                            hard_index = [hard_index[0]]
                            wrong_texts.extend([self.texts[j] for j in hard_index])
                            gold_labels.extend([self.gold_labels[j] for j in hard_index])
                            student_probs.extend([torch.mean(max_probs[j]).item() for j in hard_index])
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss.item()

        self.texts = []
        self.labels = []
        self.gold_labels = []

        if type(self.num_labels) == int:
            if len(wrong_texts) > 0:
                if self.prompt_method == 'evokd':
                    new_text, new_label = self.feedback(wrong_texts, true_labels, wrong_preds)
                    self.texts.extend(new_text)
                    self.labels.extend(new_label)
                    self.new_texts.extend(new_text)
                    self.new_labels.extend(new_label)
                elif self.prompt_method == 'dualchecker':
                    print("Feedback to teacher model")
                    print(f"received {len(wrong_texts)} samples")
                    for k in range(len(wrong_texts)):
                        new_instruction = self.feedback(wrong_texts[k], gold_labels[k])
                        self.instructions.append(new_instruction)
                        self.instructions_probs.extend(student_probs)
        else:
            if len(wrong_texts) > 0:
                wrong_inputs = [self.tokenizer.tokenize(text) for text in wrong_texts]
                categories = {'0': 0,
                              'B-cuz': 1,
                              'I-cuz': 2,
                              'B-efc': 3,
                              'I-efc': 4,
                              0: '0',
                              1: 'B-cuz',
                              2: 'I-cuz',
                              3: 'B-efc',
                              4: 'I-efc',
                              }
                if self.prompt_method == 'evokd':
                    label_entities = [split_entity([categories[i] for i in label]) for label in true_labels]
                    true_labels = [return_cause_effect_from_entity(label_entity, wrong_input) for label_entity, wrong_input in zip(label_entities, wrong_inputs)]
                    pred_entity = [split_entity([categories[i.item()] for i in pred ]) for pred in wrong_preds]
                    wrong_preds = [return_cause_effect_from_entity(pred_entity, wrong_input) for pred_entity, wrong_input in zip(pred_entity, wrong_inputs)]

                    new_text, new_label = self.feedback(wrong_texts, true_labels, wrong_preds)
                    self.texts.extend(new_text)
                    self.labels.extend(new_label)
                    self.new_texts.extend(new_text)
                    self.new_labels.extend(new_label)
                elif self.prompt_method == 'dualchecker':
                    print("Feedback to teacher model")
                    print(f"received {len(wrong_texts)} samples")
                    for k in range(len(wrong_texts)):
                        new_instruction = self.feedback(wrong_texts[k], gold_labels[k])
                        self.instructions.append(new_instruction)
                        self.instructions_probs.extend(student_probs)


    def save_model(self):
        self.student_model.save_pretrained(self.save_path)
        print(f"Model saved at {self.save_path}")


