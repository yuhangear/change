# Copyright (c) 2021 Mobvoi Inc. (authors: Di Wu)
#               2022 Tinnove Inc (authors: Wei Ren)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from textgrid import TextGrid, IntervalTier

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.ctc_util import forced_align
from wenet.utils.common import get_subsample
from wenet.utils.init_model import init_model

from abc import ABC

# import torch
# #
# from espnet.nets.scorer_interface import BatchScorerInterface

def gen_const(i,total_length):
    return 1

def gen_linear(i,total_length):
    return i

from typing import List




class Node(object):
    def __init__(self, idx, w,
                 out_nodes=None, fail_cost=0,
                 out_nodes_cost=None, fail_node= None, end_node=False,
                 father_node=None):
        if out_nodes_cost is None:
            out_nodes_cost = []
        if out_nodes is None:
            out_nodes = []
        self.idx=idx              #该节点的idx
        self.w=w               #当前节点的符号

        self.out_nodes=out_nodes   #从该节点出发，可以到达的节点
        self.out_nodes_cost=out_nodes_cost    #从该节点出发，到其它节点的cost

        self.is_end_node=end_node   #是否是重点节点

        self.fail_cost=fail_cost    #前往fail节点的cost
        self.fail_node=fail_node       #fail节点，默认指向root节点
        self.temp_total_score=0      #匹配这个单词到达这个节点的全部分数，这个在score的时候不会用，只是用来step2，计算fail 的cost
        self.father_node=father_node   #指向父节点
    def next_node(self,w):  #从这个节点出发的所有arc中，找到匹配的arc并且返回目的地节点
        #使用这函数之前需要判断arc.w是否存在，参考self.__contains__
        for i in range(len(self.out_nodes)):
            node=self.out_nodes[i]
            if node.w==w:
                return True,node,self.out_nodes_cost[i]   #匹配成功，返回下一个节点和cost

        return False,self.fail_node,self.fail_cost   #匹配失败，返回失败节点和cost

    def __contains__(self, w):
        #输出节点中是否存在w
        for i in range(len(self.out_nodes)):
            node=self.out_nodes[i]
            if node.w==w:
                return True  #匹配成功
        return False


    def __str__(self):
        outstr=str(self.idx)+":"+str(self.w)+"--->"
        for node in self.out_nodes:
            outstr+=str(node.idx)+":"+str(node.w)+" "
        outstr += "  Fail-->"+str(self.fail_node.idx)+":"+str(self.fail_node.w)+" Fcost"+str(self.fail_cost)
        return outstr

class hotword_FST(ABC):
    #hotword_list should only contain ids
    def __init__(self,hotwords,award=3,gen_score_f=gen_const):
        self.gen_score_f=gen_score_f #每匹配到一个符号，的分数增量
        self.award=award  #hotword的奖励,改变fusion weight 应该和这个效果一样。
        self.all_nodes=[Node(0, None)]  #起始节点,起始节点的return_cost为0,而且这个节点不代表任何符号，根节点没有fail_node 和 父节点

        #hotwords.sort(key=lambda x: len(x),reverse=True)  #需要从长到短添加 hotword，非常关键
        #step1 :add all hotwords and build a initial FST
        for hw in hotwords:
            self.__add_hotword(hw)
        #at this point every word have been added but all the fail path is pointing to root node
        #step2: re-route fail path
        self.__BFS_re_route(self.all_nodes[0])

        #step3: shorten the fail path
        #the FST is still usable without step3 but it will be slower
        #too difficult skip

        pass

    def score(self,state,w):
        #state是nodes的编号
        #w是当前输入符号
        node_p=self.all_nodes[state]

        #process next state
        ismatch,next_node,cost =node_p.next_node(w)
        if ismatch:   #匹配
            node_p=next_node
        else:
            node_p,cost = self.__go_through_fail_path(node_p,w)  #当前节点没有匹配，在fail_path 中搜索
                                                                #this function is very expensive and will be called twice
                                                                #some optimization can be done here?
        new_state=node_p.idx  #next state done
        # process score
        local_w=[]
        local_w_cost=[]
        for i in range(len(node_p.out_nodes)):
            local_w.append(node_p.out_nodes[i].w)     #这里score的是直接的
            local_w_cost.append(node_p.out_nodes_cost[i])

        fail_w,fail_w_cost,fail_cost=self.__score_fail_path(node_p)

        #merge to score
        for i in range(len(fail_w)):
            w=fail_w[i]
            if w in local_w:
                pass    #本地匹配的优先
            else:
                local_w.append(w)
                local_w_cost.append(fail_w_cost[i])
        return new_state, [local_w, local_w_cost], fail_cost  # 当前节点的输出弧度和对应的cost，和当前节点的返回cost
        # 值得注意的是对于hotwordFST来说所有arc的cost都是一样的


    def __score_fail_path(self,node):
        #这个函数会遍历整个fail_path，找到所有可能的next token并计算分数，
        fail_w=[]
        fail_w_cost=[]
        total_fail_cost=0
        node_p=node
        while(True):
            total_fail_cost+=node_p.fail_cost
            node_p=node_p.fail_node
            if node_p==None:
                break
            for i in range(len(node_p.out_nodes)):
                if node_p.out_nodes[i].w not in fail_w:
                    fail_w.append(node_p.out_nodes[i].w)
                    fail_w_cost.append(total_fail_cost+node_p.out_nodes_cost[i])

        return fail_w,fail_w_cost,total_fail_cost


    def __go_through_fail_path(self,node,w):
        node_p=node
        total_cost=0
        while(node_p.fail_node!=None):
            total_cost+=node_p.fail_cost
            if w in node_p.fail_node:
                ismatch,next_node,cost=node_p.fail_node.next_node(w)
                return next_node,cost+total_cost
            else:
                node_p=node_p.fail_node
        return self.all_nodes[0],total_cost
    def __BFS_re_route(self,root_node: Node):
        children_node=[root_node]
        while(len(children_node)!=0):
            temp_children_node=[]
            for node in children_node:
                self.__re_route_fail_path(node)
                temp_children_node+=node.out_nodes
            children_node=temp_children_node



    # def __re_route_fail_path(self,node: Node):
    #     w=node.w
    #     node_p=node.father_node
    #
    #     if node.idx==0 or node_p.idx==0:
    #         return  #father node or this node is root node  don't do anything
    #     if node.fail_cost == 0 and node.fail_node.idx == 0:  # fail_cost是0并且返回root节点，说明这个节点是一个结束节点，不应该被更改
    #         return
    #     while(node_p.fail_node!=None):
    #
    #         if w in node_p.fail_node:
    #             ismatch,next_node,cost=node_p.fail_node.next_node(w)
    #             node.fail_node=next_node
    #             node.fail_cost=next_node.temp_total_score-node.temp_total_score  #fail cost 需要被改变，这样顺着fail path回到部分匹配单词的时候分数才会正确
    #             return
    #         else:
    #             node_p=node_p.fail_node
    #     return #no match do nothing

    def __re_route_fail_path(self,node: Node):
        w=node.w
        node_p=node.father_node

        if node.idx==0 or node_p.idx==0:
            return  #father node or this node is root node  don't do anything
        if node.fail_cost == 0 and node.fail_node.idx == 0:  # fail_cost是0并且返回root节点，说明这个节点是一个结束节点，不应该被更改
            return
        _,_,total_return_cost=node_p.next_node(w)
        total_return_cost=-total_return_cost
        while(node_p.fail_node!=None):
            total_return_cost+=node_p.fail_cost
            if w in node_p.fail_node:
                ismatch,next_node,cost=node_p.fail_node.next_node(w)
                node.fail_node=next_node
                node.fail_cost=total_return_cost+cost  #fail cost 需要被改变，这样顺着fail path回到部分匹配单词的时候分数才会正确
                return
            else:
                node_p=node_p.fail_node
        node.fail_cost=total_return_cost
        return #no match do nothing




    def __add_hotword(self,hw):
        node_p=self.all_nodes[0]   #节点指针 首先指向起点
        score_at_each_arc=self.__gen_score(len(hw))
        for i in range(len(hw)-1):
            w=hw[i]
            match,next_node,cost=node_p.next_node(w)
            if match: #当前节点有一个输出节点与符号w一样。
                node_p=next_node
            else:
                newnode= Node(len(self.all_nodes),w,
                              fail_node=self.all_nodes[0],fail_cost=-sum(score_at_each_arc[:i + 1]),father_node=node_p) #添加新的节点，默认fail回到root节点
                newnode.temp_total_score=sum(score_at_each_arc[:i + 1]) #当前词到到当前节点累积的分数，用来在step2中计算fail的cost
                self.all_nodes.append(newnode)
                node_p.out_nodes.append(newnode)
                node_p.out_nodes_cost.append(score_at_each_arc[i])
                node_p=newnode
        w = hw[-1]
        match, next_node, cost = node_p.next_node(w)
        if match:  #这说明当前hw是之前一个hw的子词，因此当前节点,已经完成了匹配，fail不应该惩罚
            next_node.fail_cost=0
        else:
            newnode = Node(len(self.all_nodes), w,
                           fail_node=self.all_nodes[0],fail_cost=0,father_node=node_p)  # 添加新的节点，已经匹配完毕，默认fail回到root节点,不需要fail_cost
            newnode.temp_total_score = sum(score_at_each_arc)
            self.all_nodes.append(newnode)
            node_p.out_nodes.append(newnode)
            node_p.out_nodes_cost.append(score_at_each_arc[-1])

    def init_state(self,):
        return 0



    def __gen_score(self,length):
        #generate socre using self.
        score=[]
        for i in range(length):
            score.append(self.gen_score_f(i,length))

        divid=sum(score)/(self.award*length)

        for i in range(length):
            score[i]=score[i]/divid   #normalize score so that the sum of score equals to self.award*length
        return score

def generator_textgrid(maxtime, lines, output):
    # Download Praat: https://www.fon.hum.uva.nl/praat/
    interval = maxtime / (len(lines) + 1)
    margin = 0.0001

    tg = TextGrid(maxTime=maxtime)
    linetier = IntervalTier(name="line", maxTime=maxtime)

    i = 0
    for l in lines:
        s, e, w = l.split()
        linetier.add(minTime=float(s) + margin, maxTime=float(e), mark=w)

    tg.append(linetier)
    print("successfully generator {}".format(output))
    tg.write(output)


def get_frames_timestamp(alignment):
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    timestamp = []
    # get frames level duration for each token
    start = 0
    end = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1
        if end == len(alignment):
            timestamp[-1] += alignment[start:]
            break
        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1
        timestamp.append(alignment[start:end])
        start = end
    return timestamp




def get_frames_timestamp_word(alignment,mid_token):
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    timestamp = []
    # get frames level duration for each token
    start = 0
    end = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1
        if end == len(alignment):
            timestamp[-1] += alignment[start:]
            break
        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1
        timestamp.append(alignment[start:end])
        start = end
    timestamp_new=[]
    for i in timestamp[:-1]:
        i=i.copy()
        if i[-1] not in mid_token :
            timestamp_new.append(i)
        else:
            timestamp_new[-1].extend(i)


    
    last=max(timestamp[-1])

    i=timestamp[-1].copy()
    if  last not in mid_token :
        timestamp_new.append(i)
    else:
        timestamp_new[-1].extend(i)
    # timestamp
    return timestamp_new




def get_labformat(timestamp, subsample):
    begin = 0
    duration = 0
    labformat = []
    for idx, t in enumerate(timestamp):
        # 25ms frame_length,10ms hop_length, 1/subsample
        subsample = get_subsample(configs)
        # time duration
        duration = len(t) * 0.01 * subsample
        if idx < len(timestamp) - 1:
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            char_dict[t[-1]]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, char_dict[t[-1]]))
        else:
            non_blank = 0
            for i in t:
                if i != 0:
                    token = i
                    break
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            char_dict[token]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, char_dict[token]))
        begin = begin + duration
    return labformat


def get_labformat_word(timestamp, subsample):
    begin = 0
    duration = 0
    labformat = []
    for idx, t in enumerate(timestamp):
        # 25ms frame_length,10ms hop_length, 1/subsample
        subsample = get_subsample(configs)
        # time duration
        duration = len(t) * 0.01 * subsample
        if idx < len(timestamp) - 1:
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            get_word(t)))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, get_word(t)))
        else:
            non_blank = 0
            for i in t:
                if i != 0:
                    token = i
                    break
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            get_word(t)))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, get_word(t)))
        begin = begin + duration
    return labformat


def get_word(t):
    temp=[]
    for i in t:
        if i !=0 and i not in temp:
            temp.append(i)
    word=""
    for i in temp:
        word=word+char_dict[i]
    return word







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use ctc to generate alignment')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--input_file', required=True, help='format data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--non_lang_syms',
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--result_file',
                        required=True,
                        help='alignment result file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gen_praat',
                        action='store_true',
                        help='convert alignment to a praat format')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')


    parser.add_argument('--decoding_type',
                        default=None,
                        type=str,
                        help='decoding_type')
    parser.add_argument('--hot_weight',
                        type=int,
                        default=3,
                        help='hot_weight')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.batch_size > 1:
        logging.fatal('alignment mode must be running with batch_size == 1')
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    symbol_table = read_symbol_table(args.dict)
    mid_token=[]
    for k,v in symbol_table.items():
        if v!=0 and v!=1 and v!=4998 and k[0]!="▁" :
            mid_token.append(v)

    symbol_table_re=dict(zip(symbol_table.values(), symbol_table.keys()))


    # Init dataset and data loader
    ali_conf = copy.deepcopy(configs['dataset_conf'])

    ali_conf['filter_conf']['max_length'] = 102400
    ali_conf['filter_conf']['min_length'] = 0
    ali_conf['filter_conf']['token_max_length'] = 102400
    ali_conf['filter_conf']['token_min_length'] = 0
    ali_conf['filter_conf']['max_output_input_ratio'] = 102400
    ali_conf['filter_conf']['min_output_input_ratio'] = 0
    ali_conf['speed_perturb'] = False
    ali_conf['spec_aug'] = False
    ali_conf['shuffle'] = False
    ali_conf['sort'] = False
    ali_conf['fbank_conf']['dither'] = 0.0
    ali_conf['batch_conf']['batch_type'] = "static"
    ali_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    ali_dataset = Dataset(args.data_type,
                          args.input_file,
                          symbol_table,
                          ali_conf,
                          args.bpe_model,
                          non_lang_syms,
                          partition=False)

    ali_data_loader = DataLoader(ali_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)
    fst_hot=[]
    # # Read sizes regular files into a Trie
    with open("/home3/yuhang001/temp/temp/wenet/examples/gigaspeech/s0/hot_word_list") as f:
        for line in f:
            filename=line.strip()
            filename=[ int(jk) for jk in filename.split()]
            fst_hot.append(filename)

    # print(0)
    hot_weight=args.hot_weight
    print("热词权重",hot_weight)
    hFST=hotword_FST(fst_hot,award=hot_weight) 


    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w',
                               encoding='utf-8') as fout:
        for batch_idx, batch in enumerate(ali_data_loader):
            print("#" * 80)
            key, feat, target, feats_length, target_length = batch
            print(key)

            feat = feat.to(device)
            target = target.to(device)
            feats_length = feats_length.to(device)
            target_length = target_length.to(device)
            # Let's assume B = batch_size and N = beam_size
            # 1. Encoder

            # print(ctc_probs.size(1))

            #Use different decoding methods to generate alignment
            decoding_type=args.decoding_type
            print(args.decoding_type,"进行解码")
            if decoding_type =="ctc_greedy" :
                encoder_out, encoder_mask = model._forward_encoder(
                    feat, feats_length)  # (B, maxlen, encoder_dim)
                maxlen = encoder_out.size(1)
                ctc_probs = model.ctc.log_softmax(
                    encoder_out)  # (1, maxlen, vocab_size)

                ctc_probs = ctc_probs.squeeze(0)
                topk_prob, topk_index = ctc_probs.topk(1, dim=1)
                topk_index=topk_index.squeeze(-1).tolist()
                target = target.squeeze(0)
                alignment=topk_index
            elif decoding_type=="ctc_prefix_beam_search":
                hyps, _ ,ctc_probs= model._ctc_prefix_beam_search_word_boundary(feat, feats_length,
                                                    10, -1,
                                                    -1,
                                                    False)
            
                target =list(hyps[0][0])
                if len(target)==0:
                    alignment=ctc_probs.argmax(-1)
                else:
                    alignment = forced_align(ctc_probs, target)

            elif decoding_type=="attention_rescoring":
                hyps, _ ,ctc_probs= model.attention_rescoring_word_boundary(feat, feats_length,
                                                    10, -1,
                                                    -1,
                                                    False) 
            
                target =list(hyps)
                if len(target)==0:
                    alignment=ctc_probs.argmax(-1)
                else:
                    alignment = forced_align(ctc_probs, target)


            elif decoding_type=="ctc_prefix_beam_search_hot":
                print("decoding type ctc_prefix_beam_search_hot")
                hyps, _ ,ctc_probs= model._ctc_prefix_beam_search_word_boundary_hot(feat, feats_length,hFST,
                                                    10, -1,
                                                    -1,
                                                    False)
            
                target =list(hyps[0][0])
                if len(target)==0:
                    alignment=ctc_probs.argmax(-1)
                else:
                    alignment = forced_align(ctc_probs, target)

            elif decoding_type=="forced_align" :
                encoder_out, encoder_mask = model._forward_encoder(
                    feat, feats_length)  # (B, maxlen, encoder_dim)
                maxlen = encoder_out.size(1)
                ctc_probs = model.ctc.log_softmax(
                    encoder_out)  # (1, maxlen, vocab_size)
                ctc_probs = ctc_probs.squeeze(0)
                target = target.squeeze(0)
                alignment = forced_align(ctc_probs, target)
            else:
                #ctc_greedy
                encoder_out, encoder_mask = model._forward_encoder(
                    feat, feats_length)  # (B, maxlen, encoder_dim)
                maxlen = encoder_out.size(1)
                ctc_probs = model.ctc.log_softmax(
                    encoder_out)  # (1, maxlen, vocab_size)
                ctc_probs = ctc_probs.squeeze(0)
                topk_prob, topk_index = ctc_probs.topk(1, dim=1)
                topk_index=topk_index.squeeze(-1).tolist()
                target = target.squeeze(0)
                alignment=topk_index

            # alignment = forced_align(ctc_probs, target)
            print(alignment)
            fout.write('{} {}\n'.format(key[0], alignment))

            if args.gen_praat:
                if max(alignment) > 0 :
                    timestamp = get_frames_timestamp_word(alignment,mid_token)
                    print(timestamp)
                    subsample = get_subsample(configs)
                    labformat = get_labformat_word(timestamp, subsample)
                    out="".join( i.strip().split(" ")[2] for i in labformat)
                    print(out)
                    lab_path = os.path.join(os.path.dirname(args.result_file),
                                            key[0] + ".lab")
                    with open(lab_path, 'w', encoding='utf-8') as f:
                        f.writelines(labformat)

                    textgrid_path = os.path.join(os.path.dirname(args.result_file),
                                                key[0] + ".TextGrid")
                    generator_textgrid(maxtime=(len(alignment) + 1) * 0.01 *
                                    subsample,
                                    lines=labformat,
                                    output=textgrid_path)
                else:
                    timestamp = [alignment] 
                    print(timestamp)
                    subsample = get_subsample(configs)
                    labformat = get_labformat_word(timestamp, subsample) 
                    temp=labformat[0]
                    labformat[0]=temp.strip()+" <SIL> \n"
                    out="".join( i.strip().split(" ")[2] for i in labformat)
                    print(out)
                    lab_path = os.path.join(os.path.dirname(args.result_file),
                                            key[0] + ".lab")
                    with open(lab_path, 'w', encoding='utf-8') as f:
                        f.writelines(labformat)

                    textgrid_path = os.path.join(os.path.dirname(args.result_file),
                                                key[0] + ".TextGrid")
                    generator_textgrid(maxtime=(len(alignment) + 1) * 0.01 *
                                    subsample,
                                    lines=labformat,
                                    output=textgrid_path)

