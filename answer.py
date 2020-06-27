import re
from nltk.metrics import edit_distance



word_list=[]
sentence_list=[]#句子列表

#same_num=[]
answer_list=[]#识别到的文字
question_list=[]#问题
same_ans=[[] for i in range(500)]#相同答案，多行60列
same_ans_length=[[] for i in range(500)]#多行60列

 #一、读取输出的句子
def read_sentence(file_path):
    file=open(file_path)
    for lines in file:
        #word=lines.strip('\n').split(';')
        #单词间空格分开,句子分开
        sentence2=lines.replace("*","").upper().replace(";"," ").splitlines()
        sentence=lines.upper().rstrip(";").splitlines()
        #word=lines.split(' ')
        #for w in word:
            #word_list.append(w)
        for s in sentence2:
            sentence_list.append(s)
    file.close
    return sentence_list

#print(len(sentence_list))
#for i in range(20):
#    print(i)
#    print(sentence_list[i])

#a='jobsjsdflhdgaalss'#test
#b='jobsjdfylhdgaalsg'
#print(edit_distance(a,b))
#print(Levenshtein_Distance(a,b))

#二、输入问题
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

#while i<(len(sentence_list)-1):#迭代所有句子
#    d=edit_distance(sentence_list[i],sentence_list[i+1])
#    print("%s和%s的编辑距离是%d"%(sentence_list[i],sentence_list[i+1],d))
#    i+=1
#    while(d<=100):#编辑距离小的写入一类
#        same_ans.append(sentence_list[i])

#    if(d>100):
#        if len(same_ans)=1 :
#            answer=same_ans[0]        
#        else:
#            for index in range(len(same_ans)-1):#迭代所有相同答案
#                answer=same_ans[index]
#                if len(same_ans[index+1])>len(same_ans[index]):
#                    answer=same_ans[index+1]
#        answer_list.append(answer)#答案写入列表
#        same_ans.clear()#清空相同类别答案的列表

#三、迭代所有句子、答案分类
def sentence_classify(sentence_list,index):
    i=0
    j=0
    same_ans[0].append(sentence_list[0])#第一个句子写入第一行
    while i<(len(sentence_list)-1):
        d=edit_distance(sentence_list[i],sentence_list[i+1])
        #print("%s和%s的编辑距离是%d"%(sentence_list[i],sentence_list[i+1],d))#test
        #编辑距离小于index的写入一类
        if(d<=index):
            same_ans[j].append(sentence_list[i+1])#句子写入第j行
            #same_ans_length[j].append(len(sentence_list[i]))#句子的长度写入
        else:
            j+=1#另起一行
            same_ans[j].append(sentence_list[i+1])
        i+=1
    #print(same_ans)

    return same_ans,i,j


ans_length=[]#长度

#四、找到唯一答案
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

    #same_ans.clear()#清空列表释放内存
    #print(answer_list)
    #answer_list为唯一答案，各句子间的单词用空格隔开
    return answer_list

#五、删除问题单词输出“问题：答案”的形式
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
                    #sen=sen.rstrip("  ")#删除两个空格和分号
                    sentence_list.append(sen)
    
        #如果合适的答案没出现
        if min(edit_distance_list)>thr_val:
            sen="".join([que,":","-"])#“问题：-”的形式
            final_answer_list.append(sen)
        #如果合适的答案出现了
        else:
            sen=" ".join(sentence_list)#相同答案的结合
            sentence_list.clear()
            sen="".join([que,":",sen])
            final_answer_list.append(sen)

        edit_distance_list.clear()
    #ans=";".join(answer_list)
    #ans=ans.upper()
    ##ans=re.sub('[\W_]+','',ans) #删除除字母外元素
    #print(ans)
    return final_answer_list

#六、输出答案txt
def write_answer(final_answer_list):
    ans=";".join(final_answer_list)
    #ans=ans.rstrip("\n").replace(" ",";").replace(';;;',';').replace(';;',';')
    ans=ans.replace('   ',' ').replace('  ',' ').replace('\n','')
    #print(ans)
    with open("answer.txt","w") as f:
        f.write(ans)
        #for sen in final_answer_list:
        #    f.write(sen+ '\n')


def output_answer(file_name):
    #一、读取识别的句子,此处读入txt，程序中直接读入列表然后clear
    sentence_list=read_sentence(file_name)
    #print(sentence_list)
    #sentence_list=textlist
    #textlist.clear()

    #二、读取输入的问题
    question_list=read_question("./question.txt")
    #print(question_list)

    #三、找到相似的答案分类,编辑距离小的写入一类
    same_ans,i,j=sentence_classify(sentence_list,10)
    sentence_list.clear()
    #print(same_ans[0])
    print(same_ans)

    #四、找到相似答案中的唯一
    answer_list=find_only_sentence(j,same_ans)
    same_ans.clear()
    print(answer_list)

    #五、删除问题单词输出“问题：答案”的形式
    final_answer_list=output_final_answer(1,answer_list)
    #print(final_answer_list)
    answer_list.clear()
    question_list.clear()

    #六、写入txt文件
    write_answer(final_answer_list)
    final_answer_list.clear()

#output_answer("./text-yolo-pruning.txt")
#output_answer("./text.txt")

def output_answer_from_list(sentence_list):
    
    question_list=read_question("Yolo_pruning-CRNN/question.txt")
    #print(question_list)

    #三、找到相似的答案分类,编辑距离小的写入一类
    same_ans,i,j=sentence_classify(sentence_list,10)
    sentence_list.clear()
    #print(same_ans[0])
    print(same_ans)

    #四、找到相似答案中的唯一
    answer_list=find_only_sentence(j,same_ans)
    same_ans.clear()
    print(answer_list)

    #五、删除问题单词输出“问题：答案”的形式
    final_answer_list=output_final_answer(1,answer_list)
    #print(final_answer_list)
    answer_list.clear()
    question_list.clear()

    #六、写入txt文件
    write_answer(final_answer_list)
    final_answer_list.clear()

def output_answer_from_list2(sentence_list):
   
    #二、读取输入的问题
    question_list=read_question(".Yolo_pruning-CRNN/question.txt")
    #print(question_list)

    
    #五、删除问题单词输出“问题：答案”的形式
    final_answer_list=output_final_answer(1,sentence_list)
    #print(final_answer_list)
    answer_list.clear()
    question_list.clear()

    #六、写入txt文件
    write_answer(final_answer_list)
    final_answer_list.clear()