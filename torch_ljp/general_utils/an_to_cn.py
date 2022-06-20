#手写了一个阿拉伯数字转换为汉字的函数。主要用于法条标签的数值形式与刑法文本形式之间的转换。
#由于刑法文本非常规整，且仅有452条，因此没有考虑复杂的转换。

simple_map=['一','二','三','四','五','六','七','八','九']

def an2cn(an):
    """
    将阿拉伯数字转换为汉字
    输入接受int格式和str格式是阿拉伯数字
    """
    result_cn=""

    if isinstance(an,str):
        an=int(an)
    
    if int(an/100)>0:
        result_cn+=simple_map[int(an/100)-1]
        result_cn+='百'
        an%=100
    if int(an/10)>0:
        result_cn+=simple_map[int(an/10)-1]
        result_cn+='十'
        an%=10
    if an>0:
        result_cn+=simple_map[an-1]
    elif len(result_cn)==0:
        result_cn+='零'
    
    return(result_cn)
    
#print(an2cn(20))