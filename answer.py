import re
from nltk.metrics import edit_distance



word_list=[]
sentence_list=[]#句子列表

#same_num=[]
answer_list=[]#识别到的文字
question_list=[]#问题
same_ans=[[] for i in range(60)]#相同答案，多行60列
same_ans_length=[[] for i in range(60)]#多行60列

def read_sentence(file_path):
    file=open(file_path)
    for lines in file:
        sentence2=lines.replace("*","").upper().replace(";"," ").splitlines()
        sentence=lines.upper().rstrip(";").splitlines()
        for s in sentence2:
            sentence_list.append(s)
    file.close
    return sentence_list
def read_question(file_path):
    file2=open(file_path)
    for lines in file2:
        question=lines.upper().split(';')
        for q in question:
            question_list.append(q)
    question_list.pop()#去掉最后一个元素：\n
    file2.close
    #print(question_list)
    return question_list

def sentence_classify(sentence_list,index):
    i=0
    j=0
    while i<(len(sentence_list)-1):
        d=edit_distance(sentence_list[i],sentence_list[i+1])
        
        #编辑距离小于index的写入一类
        if(d<=index):
            same_ans[j].append(sentence_list[i])#句子写入第j行
        else:
            j+=1#另起一行
            same_ans[j].append(sentence_list[i])
        i+=1
    #print(same_ans)
    return same_ans,i,j


ans_length=[]#长度


def find_only_sentence(index,same_ans):
    j=index
    k=0
    while (k<=j):
        #将长度储存
        for i in range(len(same_ans[k])):
            ans_length.append(len(same_ans[k][i]))
        anx_index=ans_length.index(max(ans_length))
        answer_list.append(same_ans[k][anx_index])
        ans_length.clear()
        k+=1
    return answer_list


final_answer_list=[]#最终答案
def output_final_answer(num,answer_list):#num为编辑距离阈值
    edit_distance_list=[]
    thr_val=num#编辑距离阈值,大于此阈值就排除掉
    for que in question_list:
        for i in range(len(answer_list)):
            sen=answer_list[i]
            word=sen.split(' ')
            for w in word:
                edit_distance_list.append(edit_distance(w,que))
                if edit_distance(w,que)<=thr_val:
                    sen=sen.replace(w,'')#删除句子中跟答案一样的单词
                    sentence_list.append(sen)
        if min(edit_distance_list)>thr_val:
            sen="".join([que,":","-"])#“问题：-”的形式
            final_answer_list.append(sen)
        else:
            sen=" ".join(sentence_list)#相同答案的结合
            sentence_list.clear()
            sen="".join([que,":",sen])
            final_answer_list.append(sen)

        edit_distance_list.clear()
    return final_answer_list

def write_answer(final_answer_list):
    ans=";".join(final_answer_list)
    ans=ans.rstrip("\n").replace(" ",";").replace(';;;',';').replace(';;',';')
    with open("answer.txt","w") as f:
        f.write(ans)


def output_answer(file_name):
    sentence_list=read_sentence(file_name)
    question_list=read_question("./question.txt")
    same_ans,i,j=sentence_classify(sentence_list,30)
    sentence_list.clear()
    answer_list=find_only_sentence(j,same_ans)
    same_ans.clear()
    final_answer_list=output_final_answer(1,answer_list)
    answer_list.clear()
    question_list.clear()
    write_answer(final_answer_list)
    final_answer_list.clear()
