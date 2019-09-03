import tensorflow as tf

# 查看静态图.pb节点信息:
"""FIND GRAPH INFO"""
# tf_model_path = "D:/gitSpacePycharm/cc0output/mobilenet_2019_08_12_frozen.pb"
# tf_model_path = "D:/gitSpacePycharm/cc0output/bak1/inception_v3_2016_08_28_frozen.pb"

tf_model_path = "tmp2/frozen_inference_graph.pb"
with open(tf_model_path , 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name ='')
    ops = g.get_operations()
    N = len(ops)
    for i in [0,1,2,N-3,N-2,N-1]: # for循环设置输出的节点信息
        print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type))
        print('input(s):')
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape()))
        print('\noutput(s):'),
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape()))