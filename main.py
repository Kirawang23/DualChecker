import os
import argparse
import logging
from script.run import run
import time

class Logger(object):
    def __init__(self, filename, level='info'):
        level = logging.INFO if level == 'info' else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        self.logger.setLevel(level)
        th = logging.FileHandler(filename, 'w')
        self.logger.addHandler(th)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default='ce',
                        type=str,
                        help="task to test: cls/ce/path")
    parser.add_argument("--part",
                        default='all',
                        type=str,
                        help="test teacher or student: simi/teacher/student/all")
    parser.add_argument("--prompt_method",
                        default='dualchecker',
                        type=str,
                        help="prompt method: cot/dtot/evokd/dualchecker")
    parser.add_argument("--n_shot",
                        default='5_shot',
                        type=str,
                        help="n_shot: 0_shot/1_shot/3_shot/5_shot")
    parser.add_argument("--model_name",
                        default='gpt3.5turbo',
                        type=str,
                        help="The model to use: gpt3.5turbo/llama2")
    parser.add_argument("--student_path",
                        default='model/patent_roberta',
                        type=str,
                        help="The student model path")
    parser.add_argument("--simimodel_path",
                        default='model/SROBERTA',
                        type=str,
                        help="The student model path")
    parser.add_argument("--reprompt",
                        default=False,
                        type=bool,
                        help="Whether to reprompt for low confidence text")
    parser.add_argument("--train_ratio",
                        default=0.8,
                        type=float,
                        help="train ratio.")
    parser.add_argument("--batch_size",
                        default= 8,
                        type=int,
                        help="batch size.")
    parser.add_argument('--epoch',
                        type=int,
                        default=10,
                        help='Number of epochs')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-5,
                        help='Learning rate')
    parser.add_argument('--eval_step',
                        type=int,
                        default=100,
                        help='Evaluation step')
    parser.add_argument('--max_length',
                        type=int,
                        default=512,
                        help='Max length')
    parser.add_argument('--gpus',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='GPU ids')
    parser.add_argument('--use_gpu',
                        type=int,
                        default=1,
                        help='Use GPU or not')
    parser.add_argument("--teacher_threshold",
                        default=0.85,
                        type=float,
                        help="threshold for teacher")
    parser.add_argument("--student_threshold",
                        default=0.6,
                        type=float,
                        help="threshold for student")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed.")
    parser.add_argument("--log_dir",
                        default='log/',
                        type=str,
                        help="Path for Logging file")
    parser.add_argument("--save_path",
                        default='save_model/',
                        type=str,
                        help="Path for saving model")
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if not os.path.exists(args.log_dir + args.task_name):
        os.makedirs(args.log_dir + args.task_name)

    if not os.path.exists(args.save_path + args.task_name):
        os.makedirs(args.save_path + args.task_name)

    log = Logger(
        args.log_dir + args.task_name + '/' + args.part + '_' + args.prompt_method + '_' + args.n_shot + '_' + args.model_name + '_' + str(
            args.batch_size) + '_' + str(args.teacher_threshold) + '_'  + str(args.student_threshold) + '_'+ str(args.seed) + '.log')

    if args.task_name == 'cls':
        args.data_dir = 'data/cls_data.json'
    else:
        args.data_dir = 'data/hard_cls_data.json'

    if 'gpt' in args.model_name:
        args.model_type = 'close'
    else:
        args.model_type = 'open'

    start = time.time()
    log.logger.info(f'************************Start Test************************')
    log.logger.info(
        f"【task】: {args.task_name} 【part】: {args.part} 【prompt_method】: {args.prompt_method} 【model_name】: {args.model_name} 【n_shot】: {args.n_shot} 【task_name】: {args.task_name} 【batch size】: {args.batch_size} 【teacher_threshold】: {args.teacher_threshold} 【student_threshold】: {args.student_threshold} 【seed】: {args.seed}")

    run(args, log)

    end = time.time()
    log.logger.info(f'************************End Test************************')
    log.logger.info("Processing time: {}".format(end - start))
    print(
        f'save log to {args.log_dir + args.task_name + "/" + args.part + "_" + args.prompt_method + "_" + args.n_shot + "_" + args.model_name  + "_" + str(args.batch_size) + "_" + str(args.teacher_threshold) + "_" + str(args.student_threshold) + "_" + str(args.seed) + ".log"}')

if __name__ == "__main__":
    main()