Ü)
ĹŻ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
ź
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ý
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


FakeQuantWithMinMaxArgs

inputs
outputs"
minfloat%  ŔŔ"
maxfloat%  Ŕ@"
num_bitsint"
narrow_rangebool( 
s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%ˇŃ8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
=
Mul
x"T
y"T
z"T"
Ttype:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu6
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
:
Sub
x"T
y"T
z"T"
Ttype:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
*1.13.12unknown8Ř!

hub_input/imagesPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕŕ*&
shape:˙˙˙˙˙˙˙˙˙ŕŕ
T
hub_input/Mul/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
s
hub_input/MulMulhub_input/imageshub_input/Mul/y*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕŕ*
T0

!hub_input/FakeQuantWithMinMaxArgsFakeQuantWithMinMaxArgshub_input/Mul*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕŕ*
min%    *
max%b @
T
hub_input/Sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

hub_input/SubSub!hub_input/FakeQuantWithMinMaxArgshub_input/Sub/y*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕŕ*
T0

#hub_input/FakeQuantWithMinMaxArgs_1FakeQuantWithMinMaxArgshub_input/Sub*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕŕ*
min%  ż*
max%Ĺ ?
É
?MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*
_output_shapes
:*%
valueB"             
´
>MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/meanConst*
valueB
 *    */
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*
_output_shapes
: 

IMobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/shape*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*&
_output_shapes
: *
T0
ł
=MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/mulMulIMobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/TruncatedNormal@MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*&
_output_shapes
: 
Ą
9MobilenetV1/Conv2d_0/weights/Initializer/truncated_normalAdd=MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/mul>MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal/mean*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*&
_output_shapes
: *
T0
Í
MobilenetV1/Conv2d_0/weightsVarHandleOp*
shape: */
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*-
shared_nameMobilenetV1/Conv2d_0/weights*
_output_shapes
: 

=MobilenetV1/Conv2d_0/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpMobilenetV1/Conv2d_0/weights*
_output_shapes
: 
Î
#MobilenetV1/Conv2d_0/weights/AssignAssignVariableOpMobilenetV1/Conv2d_0/weights9MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0
Ć
0MobilenetV1/Conv2d_0/weights/Read/ReadVariableOpReadVariableOpMobilenetV1/Conv2d_0/weights*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*&
_output_shapes
: 

HMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
˝
XMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpMobilenetV1/Conv2d_0/weights*
dtype0*&
_output_shapes
: 
Î
IMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2LossL2LossXMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0
˙
BMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizerMulHMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/scaleIMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

.MobilenetV1/MobilenetV1/Conv2d_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      

6MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOpReadVariableOpMobilenetV1/Conv2d_0/weights*
dtype0*&
_output_shapes
: 
đ
'MobilenetV1/MobilenetV1/Conv2d_0/Conv2DConv2D#hub_input/FakeQuantWithMinMaxArgs_16MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp *
T0*
strides
*
paddingSAME
ť
5MobilenetV1/Conv2d_0/BatchNorm/gamma/Initializer/onesConst*
valueB *  ?*7
_class-
+)loc:@MobilenetV1/Conv2d_0/BatchNorm/gamma*
dtype0*
_output_shapes
: 
Ů
$MobilenetV1/Conv2d_0/BatchNorm/gammaVarHandleOp*
shape: *7
_class-
+)loc:@MobilenetV1/Conv2d_0/BatchNorm/gamma*
dtype0*
_output_shapes
: *5
shared_name&$MobilenetV1/Conv2d_0/BatchNorm/gamma

EMobilenetV1/Conv2d_0/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp$MobilenetV1/Conv2d_0/BatchNorm/gamma*
_output_shapes
: 
â
+MobilenetV1/Conv2d_0/BatchNorm/gamma/AssignAssignVariableOp$MobilenetV1/Conv2d_0/BatchNorm/gamma5MobilenetV1/Conv2d_0/BatchNorm/gamma/Initializer/ones*7
_class-
+)loc:@MobilenetV1/Conv2d_0/BatchNorm/gamma*
dtype0
Ň
8MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp$MobilenetV1/Conv2d_0/BatchNorm/gamma*7
_class-
+)loc:@MobilenetV1/Conv2d_0/BatchNorm/gamma*
dtype0*
_output_shapes
: 
ş
5MobilenetV1/Conv2d_0/BatchNorm/beta/Initializer/zerosConst*
valueB *    *6
_class,
*(loc:@MobilenetV1/Conv2d_0/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ö
#MobilenetV1/Conv2d_0/BatchNorm/betaVarHandleOp*
shape: *6
_class,
*(loc:@MobilenetV1/Conv2d_0/BatchNorm/beta*
dtype0*
_output_shapes
: *4
shared_name%#MobilenetV1/Conv2d_0/BatchNorm/beta

DMobilenetV1/Conv2d_0/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp#MobilenetV1/Conv2d_0/BatchNorm/beta*
_output_shapes
: 
ß
*MobilenetV1/Conv2d_0/BatchNorm/beta/AssignAssignVariableOp#MobilenetV1/Conv2d_0/BatchNorm/beta5MobilenetV1/Conv2d_0/BatchNorm/beta/Initializer/zeros*6
_class,
*(loc:@MobilenetV1/Conv2d_0/BatchNorm/beta*
dtype0
Ď
7MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOpReadVariableOp#MobilenetV1/Conv2d_0/BatchNorm/beta*
_output_shapes
: *6
_class,
*(loc:@MobilenetV1/Conv2d_0/BatchNorm/beta*
dtype0
Č
<MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *=
_class3
1/loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ë
*MobilenetV1/Conv2d_0/BatchNorm/moving_meanVarHandleOp*
shape: *=
_class3
1/loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *;
shared_name,*MobilenetV1/Conv2d_0/BatchNorm/moving_mean
Ľ
KMobilenetV1/Conv2d_0/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp*MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
_output_shapes
: 
ű
1MobilenetV1/Conv2d_0/BatchNorm/moving_mean/AssignAssignVariableOp*MobilenetV1/Conv2d_0/BatchNorm/moving_mean<MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Initializer/zeros*=
_class3
1/loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
dtype0
ä
>MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp*MobilenetV1/Conv2d_0/BatchNorm/moving_mean*=
_class3
1/loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
Ď
?MobilenetV1/Conv2d_0/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
: *
valueB *  ?*A
_class7
53loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
dtype0
÷
.MobilenetV1/Conv2d_0/BatchNorm/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
shape: *A
_class7
53loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_variance
­
OMobilenetV1/Conv2d_0/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
_output_shapes
: 

5MobilenetV1/Conv2d_0/BatchNorm/moving_variance/AssignAssignVariableOp.MobilenetV1/Conv2d_0/BatchNorm/moving_variance?MobilenetV1/Conv2d_0/BatchNorm/moving_variance/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
dtype0
đ
BMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_0/BatchNorm/moving_variance*A
_class7
53loc:@MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOpReadVariableOp$MobilenetV1/Conv2d_0/BatchNorm/gamma*
dtype0*
_output_shapes
: 

;MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1ReadVariableOp#MobilenetV1/Conv2d_0/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ż
HMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp*MobilenetV1/Conv2d_0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ľ
JMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
ů
9MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormFusedBatchNorm'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D9MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp;MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_1HMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOpJMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙pp : : : : *
T0
u
0MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Wě?

&MobilenetV1/MobilenetV1/Conv2d_0/Relu6Relu6)MobilenetV1/MobilenetV1/Conv2d_0/add_fold*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp *
T0
ń
SMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"             *C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0
Ü
RMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *
valueB
 *    
Ţ
TMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Č
]MobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*&
_output_shapes
: 

QMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*&
_output_shapes
: *
T0*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights
ń
MMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal/mean*&
_output_shapes
: *
T0*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights

0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsVarHandleOp*
shape: *C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*A
shared_name20MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
_output_shapes
: 
ą
QMobilenetV1/Conv2d_1_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_1_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsMMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_1_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*&
_output_shapes
: 
ź
CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
dtype0*&
_output_shapes
: 

:MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:

BMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
¸
4MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseDepthwiseConv2dNativeBMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOp*
T0*
strides
*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 
Ď
?MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB *  ?*A
_class7
53loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: 
÷
.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gammaVarHandleOp*
shape: *A
_class7
53loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
dtype0
đ
BMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
_output_shapes
: *A
_class7
53loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
dtype0
Î
?MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB *    *@
_class6
42loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
ô
-MobilenetV1/Conv2d_1_depthwise/BatchNorm/betaVarHandleOp*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
shape: *@
_class6
42loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
dtype0
í
AMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ü
FMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *G
_class=
;9loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 

4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_meanVarHandleOp*
shape: *G
_class=
;9loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ă
IMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB *  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 

8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_varianceVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Ž
CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: 
Ż
EMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ă
RMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
É
TMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
¸
CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙pp : : : : 

:MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
¨
0MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_1_depthwise/add_fold*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp *
T0
Ý
IMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   *9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights
Č
HMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*
_output_shapes
: 
Ş
SMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/shape*&
_output_shapes
: @*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0
Ű
GMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/stddev*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*&
_output_shapes
: @*
T0
É
CMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal/mean*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*&
_output_shapes
: @*
T0
ë
&MobilenetV1/Conv2d_1_pointwise/weightsVarHandleOp*
dtype0*
_output_shapes
: *7
shared_name(&MobilenetV1/Conv2d_1_pointwise/weights*
shape: @*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights

GMobilenetV1/Conv2d_1_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_1_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_1_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_1_pointwise/weightsCMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0
ä
:MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_1_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*&
_output_shapes
: @

RMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ń
bMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*&
_output_shapes
: @
â
SMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ż
@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*&
_output_shapes
: @
­
1MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@*
T0*
strides
*
paddingSAME
Ď
?MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Initializer/onesConst*
valueB@*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
:@
÷
.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gammaVarHandleOp*
shape:@*A
_class7
53loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
dtype0
đ
BMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
_output_shapes
:@*A
_class7
53loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
dtype0
Î
?MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *@
_class6
42loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta
ô
-MobilenetV1/Conv2d_1_pointwise/BatchNorm/betaVarHandleOp*@
_class6
42loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
shape:@
Ť
NMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
dtype0
í
AMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
:@
Ü
FMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*
valueB@*    

4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
shape:@*G
_class=
;9loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
dtype0
š
UMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@
ă
IMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB@*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@

8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
shape:@*K
_classA
?=loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
dtype0
Á
YMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
Ž
CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma*
_output_shapes
:@*
dtype0
Ż
EMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
:@
Ă
RMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@
É
TMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
ľ
CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙pp@:@:@:@:@*
T0

:MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
¨
0MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_1_pointwise/add_fold*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@
ń
SMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      @      *C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Č
]MobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*&
_output_shapes
:@

QMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*&
_output_shapes
:@
ń
MMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*&
_output_shapes
:@

0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsVarHandleOp*
shape:@*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_2_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_2_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_2_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsMMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_2_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*&
_output_shapes
:@
ź
CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*&
_output_shapes
:@

:MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      

BMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Â
4MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOp*
T0*
strides
*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@
Ď
?MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB@*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
:@
÷
.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gammaVarHandleOp*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
shape:@*A
_class7
53loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
dtype0
đ
BMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
:@
Î
?MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB@*    *@
_class6
42loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
:@
ô
-MobilenetV1/Conv2d_2_depthwise/BatchNorm/betaVarHandleOp*
shape:@*@
_class6
42loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
dtype0
í
AMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
:@
Ü
FMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*
valueB@*    

4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:@*G
_class=
;9loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*G
_class=
;9loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean
ă
IMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*
valueB@*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance

8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_varianceVarHandleOp*K
_classA
?=loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
shape:@
Á
YMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
Ž
CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
:@
Ż
EMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
:@
Ă
RMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@
É
TMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
¸
CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
epsilon%o:*G
_output_shapes5
3:˙˙˙˙˙˙˙˙˙88@:@:@:@:@*
T0*
is_training( 

:MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
¨
0MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_2_depthwise/add_fold*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@*
T0
Ý
IMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"      @      *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*
_output_shapes
: 
Ť
SMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/shape*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*'
_output_shapes
:@*
T0
Ü
GMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*'
_output_shapes
:@
Ę
CMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal/mean*'
_output_shapes
:@*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
ě
&MobilenetV1/Conv2d_2_pointwise/weightsVarHandleOp*
shape:@*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*
_output_shapes
: *7
shared_name(&MobilenetV1/Conv2d_2_pointwise/weights

GMobilenetV1/Conv2d_2_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_2_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_2_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_2_pointwise/weightsCMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0
ĺ
:MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_2_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*'
_output_shapes
:@

RMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *ŹĹ'8*
dtype0
Ň
bMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*'
_output_shapes
:@
â
SMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_2_pointwise/weights*'
_output_shapes
:@*
dtype0
Ž
1MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gammaVarHandleOp*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
dtype0
­
OMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_2_pointwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Initializer/zeros*
dtype0*@
_class6
42loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta
î
AMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_meanVarHandleOp*E
shared_name64MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
dtype0
š
UMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ä
RMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙88::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ConstConst*
_output_shapes
: *
valueB
 *Wě?*
dtype0
Š
0MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_2_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0
ń
SMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0
Ţ
TMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
É
]MobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/shape*'
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0

QMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*'
_output_shapes
:
ň
MMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*'
_output_shapes
:

0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_3_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_3_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_3_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsMMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights
˝
CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_3_depthwise/depthwise_weights*'
_output_shapes
:*
dtype0

:MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

BMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ă
4MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
strides
*
T0*
paddingSAME
Ń
?MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
dtype0*?
shared_name0.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
_output_shapes
: 
­
OMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
_output_shapes	
:*A
_class7
53loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
dtype0
Đ
?MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_3_depthwise/BatchNorm/betaVarHandleOp*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
_output_shapes	
:*@
_class6
42loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
dtype0
Ţ
FMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*I
shared_name:8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙88::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ConstConst*
_output_shapes
: *
valueB
 *Wě?*
dtype0
Š
0MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_3_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
Ý
IMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
Č
HMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*
_output_shapes
: 
Ź
SMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/shape*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*(
_output_shapes
:
Ý
GMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal/mean*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*(
_output_shapes
:*
T0
í
&MobilenetV1/Conv2d_3_pointwise/weightsVarHandleOp*
_output_shapes
: *7
shared_name(&MobilenetV1/Conv2d_3_pointwise/weights*
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0

GMobilenetV1/Conv2d_3_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_3_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_3_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_3_pointwise/weightsCMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_3_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*(
_output_shapes
:

RMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0

LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ą
@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
dtype0
ő
-MobilenetV1/Conv2d_3_pointwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
dtype0
š
UMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*G
_class=
;9loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean
ĺ
IMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*
valueB*  ?

8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_varianceVarHandleOp*K
_classA
?=loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
shape:
Á
YMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ş
CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙88::::*
T0*
is_training( 

:MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_3_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0
ń
SMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
Ü
RMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
É
]MobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*
T0

QMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*'
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
ň
MMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*'
_output_shapes
:*
T0

0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_4_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_4_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_4_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsMMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_4_depthwise/depthwise_weights*'
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0
˝
CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_4_depthwise/depthwise_weights*'
_output_shapes
:*
dtype0

:MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

BMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ă
4MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
dtype0*?
shared_name0.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
_output_shapes
: 
­
OMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Initializer/ones*
dtype0*A
_class7
53loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma
ń
BMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:*
valueB*    
ő
-MobilenetV1/Conv2d_4_depthwise/BatchNorm/betaVarHandleOp*@
_class6
42loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
shape:
Ť
NMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
_output_shapes	
:*@
_class6
42loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
dtype0
Ţ
FMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*G
_class=
;9loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean
ĺ
IMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma*
_output_shapes	
:*
dtype0
°
EMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

:MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_4_depthwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ý
IMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*
_output_shapes
: 
Ź
SMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/shape*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*(
_output_shapes
:
Ý
GMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal/mean*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*(
_output_shapes
:*
T0
í
&MobilenetV1/Conv2d_4_pointwise/weightsVarHandleOp*
dtype0*
_output_shapes
: *7
shared_name(&MobilenetV1/Conv2d_4_pointwise/weights*
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights

GMobilenetV1/Conv2d_4_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_4_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_4_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_4_pointwise/weightsCMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*(
_output_shapes
:*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights

RMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0

LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ą
@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_4_pointwise/weights*(
_output_shapes
:*
dtype0
Ž
1MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
T0*
paddingSAME
Ń
?MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma
ř
.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gammaVarHandleOp*
dtype0*?
shared_name0.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
_output_shapes
: *
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_4_pointwise/BatchNorm/betaVarHandleOp*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*
valueB*    

4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_meanVarHandleOp*G
_class=
;9loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
dtype0*E
shared_name64MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:
š
UMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
dtype0

8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
dtype0*I
shared_name:8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
Á
YMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
Ę
TMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_4_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
SMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0
É
]MobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

QMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*'
_output_shapes
:
ň
MMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*'
_output_shapes
:*
T0

0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_5_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_5_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_5_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsMMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_5_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
˝
CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

:MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0

BMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ă
4MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOp*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma
ř
.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gammaVarHandleOp*A
_class7
53loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
dtype0*?
shared_name0.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
_output_shapes
: *
shape:
­
OMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:*
valueB*    
ő
-MobilenetV1/Conv2d_5_depthwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
dtype0*>
shared_name/-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
_output_shapes
: 
Ť
NMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_meanVarHandleOp*G
_class=
;9loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
shape:
š
UMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance

8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*I
shared_name:8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*K
_classA
?=loc:@MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance
Ż
CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_5_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
IMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*
_output_shapes
: 
Ź
SMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/shape*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*(
_output_shapes
:
Ý
GMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal/mean*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*(
_output_shapes
:*
T0
í
&MobilenetV1/Conv2d_5_pointwise/weightsVarHandleOp*
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*7
shared_name(&MobilenetV1/Conv2d_5_pointwise/weights*
_output_shapes
: 

GMobilenetV1/Conv2d_5_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_5_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_5_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_5_pointwise/weightsCMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*(
_output_shapes
:*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights

RMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_5_pointwise/weights*(
_output_shapes
:*
dtype0
â
SMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ą
@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_5_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
dtype0
ř
.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta
ő
-MobilenetV1/Conv2d_5_pointwise/BatchNorm/betaVarHandleOp*>
shared_name/-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
_output_shapes
: *
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
dtype0
Ť
NMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
dtype0*E
shared_name64MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
š
UMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*I
shared_name:8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
_output_shapes	
:*K
_classA
?=loc:@MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
dtype0
Ż
CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_5_pointwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
SMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
É
]MobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/shape*'
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0

QMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*'
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights
ň
MMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*'
_output_shapes
:

0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*A
shared_name20MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
_output_shapes
: 
ą
QMobilenetV1/Conv2d_6_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_6_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsMMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_6_depthwise/depthwise_weights*'
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0
˝
CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

:MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

BMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ă
4MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOp*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma
ř
.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gammaVarHandleOp*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_6_depthwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
dtype0

4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_meanVarHandleOp*E
shared_name64MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
dtype0
š
UMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
_output_shapes	
:*K
_classA
?=loc:@MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
dtype0
Ż
CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
Ę
TMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_6_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
IMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0
Ę
JMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*
_output_shapes
: 
Ź
SMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/shape*(
_output_shapes
:*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0
Ý
GMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights
í
&MobilenetV1/Conv2d_6_pointwise/weightsVarHandleOp*
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*7
shared_name(&MobilenetV1/Conv2d_6_pointwise/weights*
_output_shapes
: 

GMobilenetV1/Conv2d_6_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_6_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_6_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_6_pointwise/weightsCMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_6_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*(
_output_shapes
:

RMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ą
@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:*
valueB*  ?
ř
.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
_output_shapes	
:*A
_class7
53loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
dtype0
Đ
?MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_6_pointwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_meanVarHandleOp*
dtype0*E
shared_name64MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_varianceVarHandleOp*I
shared_name:8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
dtype0
Á
YMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance*
_output_shapes	
:*
dtype0
ş
CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_6_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
SMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
É
]MobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

QMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*'
_output_shapes
:*
T0
ň
MMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal/mean*'
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights

0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_7_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_7_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_7_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsMMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_7_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
˝
CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

:MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

BMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ă
4MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma
ř
.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
dtype0*?
shared_name0.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
_output_shapes
: 
­
OMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
dtype0
ő
-MobilenetV1/Conv2d_7_depthwise/BatchNorm/betaVarHandleOp*
dtype0*>
shared_name/-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
_output_shapes
: *
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
dtype0

4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_meanVarHandleOp*E
shared_name64MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
dtype0
š
UMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
dtype0

8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_7_depthwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ý
IMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*
_output_shapes
: 
Ę
JMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
Ź
SMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
Ý
GMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal/mean*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*(
_output_shapes
:*
T0
í
&MobilenetV1/Conv2d_7_pointwise/weightsVarHandleOp*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*
_output_shapes
: *7
shared_name(&MobilenetV1/Conv2d_7_pointwise/weights*
shape:

GMobilenetV1/Conv2d_7_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_7_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_7_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_7_pointwise/weightsCMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_7_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*(
_output_shapes
:

RMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0

LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ą
@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_7_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
T0*
paddingSAME
Ń
?MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma
ř
.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Initializer/ones*
dtype0*A
_class7
53loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma
ń
BMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_7_pointwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
dtype0*>
shared_name/-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
_output_shapes
: 
Ť
NMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
dtype0*E
shared_name64MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
š
UMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*G
_class=
;9loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean
ĺ
IMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ä
RMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ConstConst*
_output_shapes
: *
valueB
 *Wě?*
dtype0
Š
0MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_7_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
SMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ţ
TMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
É
]MobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*
T0

QMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:
ň
MMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:*
T0

0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsVarHandleOp*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_8_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_8_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsMMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_8_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
˝
CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:*
dtype0

:MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

BMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ă
4MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
dtype0
ő
-MobilenetV1/Conv2d_8_depthwise/BatchNorm/betaVarHandleOp*@
_class6
42loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
shape:
Ť
NMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*
valueB*    

4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_varianceVarHandleOp*I
shared_name:8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
dtype0
Á
YMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Initializer/ones*
dtype0*K
_classA
?=loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance

LMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*K
_classA
?=loc:@MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance
Ż
CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_8_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
IMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0
Č
HMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/meanConst*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*
_output_shapes
: *
valueB
 *    
Ę
JMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*
_output_shapes
: 
Ź
SMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/shape*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*(
_output_shapes
:
Ý
GMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal/mean*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*(
_output_shapes
:
í
&MobilenetV1/Conv2d_8_pointwise/weightsVarHandleOp*7
shared_name(&MobilenetV1/Conv2d_8_pointwise/weights*
_output_shapes
: *
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0

GMobilenetV1/Conv2d_8_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_8_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_8_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_8_pointwise/weightsCMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_8_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*(
_output_shapes
:

RMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0

LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ą
@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp*
strides
*
T0*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
?MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gammaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta
ő
-MobilenetV1/Conv2d_8_pointwise/BatchNorm/betaVarHandleOp*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
dtype0*E
shared_name64MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
š
UMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance
Á
YMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
_output_shapes	
:*K
_classA
?=loc:@MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
dtype0
Ż
CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_8_pointwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
SMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ü
RMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *
valueB
 *    
Ţ
TMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *
valueB
 *ěQ¸=
É
]MobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*
T0

QMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul]MobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*'
_output_shapes
:
ň
MMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normalAddQMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/mulRMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*'
_output_shapes
:

0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsVarHandleOp*
shape:*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *A
shared_name20MobilenetV1/Conv2d_9_depthwise/depthwise_weights
ą
QMobilenetV1/Conv2d_9_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp0MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
_output_shapes
: 

7MobilenetV1/Conv2d_9_depthwise/depthwise_weights/AssignAssignVariableOp0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsMMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0

DMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_9_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
˝
CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOpReadVariableOp0MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

:MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

BMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ă
4MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVarsCMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ń
?MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
dtype0
ř
.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gammaVarHandleOp*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma?MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_9_depthwise/BatchNorm/betaVarHandleOp*>
shared_name/-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
_output_shapes
: *
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
dtype0
Ť
NMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta?MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
dtype0
î
AMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*@
_class6
42loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ţ
FMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*K
_classA
?=loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_varianceVarHandleOp*
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
dtype0*I
shared_name:8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
Á
YMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
°
EMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
Ę
TMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
˝
CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm4MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseCMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

:MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Wě?
Š
0MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_9_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
IMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*
_output_shapes
:
Č
HMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/meanConst*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*
_output_shapes
: *
valueB
 *    
Ę
JMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ěQ¸=*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights
Ź
SMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalIMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/shape*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*(
_output_shapes
:
Ý
GMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/mulMulSMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/TruncatedNormalJMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/stddev*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*(
_output_shapes
:
Ë
CMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normalAddGMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/mulHMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal/mean*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*(
_output_shapes
:
í
&MobilenetV1/Conv2d_9_pointwise/weightsVarHandleOp*
dtype0*7
shared_name(&MobilenetV1/Conv2d_9_pointwise/weights*
_output_shapes
: *
shape:*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights

GMobilenetV1/Conv2d_9_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp&MobilenetV1/Conv2d_9_pointwise/weights*
_output_shapes
: 
ö
-MobilenetV1/Conv2d_9_pointwise/weights/AssignAssignVariableOp&MobilenetV1/Conv2d_9_pointwise/weightsCMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0
ć
:MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_9_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*(
_output_shapes
:

RMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'8*
dtype0*
_output_shapes
: 
Ó
bMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*(
_output_shapes
:
â
SMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LossbMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0

LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizerMulRMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/scaleSMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ą
@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOpReadVariableOp&MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*(
_output_shapes
:
Ž
1MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DConv2DLMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides

Ń
?MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*A
_class7
53loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gammaVarHandleOp*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma
­
OMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
_output_shapes
: 

5MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/AssignAssignVariableOp.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma?MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Initializer/ones*A
_class7
53loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
dtype0
ń
BMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*A
_class7
53loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Đ
?MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ő
-MobilenetV1/Conv2d_9_pointwise/BatchNorm/betaVarHandleOp*
shape:*@
_class6
42loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *>
shared_name/-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
Ť
NMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
_output_shapes
: 

4MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/AssignAssignVariableOp-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta?MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Initializer/zeros*
dtype0*@
_class6
42loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
î
AMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:*@
_class6
42loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
Ţ
FMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_meanVarHandleOp*
shape:*G
_class=
;9loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *E
shared_name64MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean
š
UMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
Ł
;MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_meanFMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
dtype0

HMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*G
_class=
;9loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ĺ
IMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*
valueB*  ?

8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_varianceVarHandleOp*I
shared_name:8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*K
_classA
?=loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
dtype0
Á
YMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
˛
?MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_varianceIMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Initializer/ones*K
_classA
?=loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
dtype0

LMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*K
_classA
?=loc:@MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ż
CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma*
_output_shapes	
:*
dtype0
°
EMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp-MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ä
RMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp4MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ę
TMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp8MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ş
CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm1MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DCMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOpEMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1RMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpTMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

:MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Š
0MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6Relu63MobilenetV1/MobilenetV1/Conv2d_9_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
TMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"            *D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0
Ţ
SMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
ŕ
UMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ě
^MobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/shape*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*
T0

RMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul^MobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalUMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*'
_output_shapes
:*
T0
ö
NMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normalAddRMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/mulSMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal/mean*'
_output_shapes
:*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights

1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsVarHandleOp*
shape:*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*B
shared_name31MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
_output_shapes
: 
ł
RMobilenetV1/Conv2d_10_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp1MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
_output_shapes
: 
˘
8MobilenetV1/Conv2d_10_depthwise/depthwise_weights/AssignAssignVariableOp1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsNMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0

EMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_10_depthwise/depthwise_weights*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
ż
DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

;MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

CMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ĺ
5MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVarsDMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ó
@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*B
_class8
64loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ű
/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gammaVarHandleOp*
dtype0*@
shared_name1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
_output_shapes
: *
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ň
@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_10_depthwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta
­
OMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ŕ
GMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *H
_class>
<:loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_meanVarHandleOp*
dtype0*F
shared_name75MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean
ť
VMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Initializer/zeros*
dtype0*H
_class>
<:loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean

IMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean
ç
JMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Initializer/onesConst*L
_classB
@>loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*
valueB*  ?

9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance
Ă
ZMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ă
DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm5MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

;MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Ť
1MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_10_depthwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
JMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
dtype0*
_output_shapes
:
Ę
IMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
Ě
KMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
dtype0*
_output_shapes
: 
Ż
TMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalJMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/shape*(
_output_shapes
:*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
dtype0
á
HMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/mulMulTMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/TruncatedNormalKMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/stddev*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*(
_output_shapes
:
Ď
DMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normalAddHMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/mulIMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal/mean*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*(
_output_shapes
:
đ
'MobilenetV1/Conv2d_10_pointwise/weightsVarHandleOp*
dtype0*
_output_shapes
: *8
shared_name)'MobilenetV1/Conv2d_10_pointwise/weights*
shape:*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights

HMobilenetV1/Conv2d_10_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'MobilenetV1/Conv2d_10_pointwise/weights*
_output_shapes
: 
ú
.MobilenetV1/Conv2d_10_pointwise/weights/AssignAssignVariableOp'MobilenetV1/Conv2d_10_pointwise/weightsDMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
dtype0
é
;MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_10_pointwise/weights*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
dtype0*(
_output_shapes
:

SMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *ŹĹ'8
Ő
cMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_10_pointwise/weights*
dtype0*(
_output_shapes
:
ä
TMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LosscMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0
 
MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizerMulSMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/scaleTMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
ł
AMobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_10_pointwise/weights*
dtype0*(
_output_shapes
:
ą
2MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DConv2DMMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVarsAMobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ó
@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Initializer/onesConst*B
_class8
64loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:*
valueB*  ?
ű
/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gammaVarHandleOp*
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:*B
_class8
64loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma
Ň
@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta
ř
.MobilenetV1/Conv2d_10_pointwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
dtype0*?
shared_name0.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
_output_shapes
: 
­
OMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Initializer/zeros*
dtype0*A
_class7
53loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta
ń
BMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ŕ
GMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *H
_class>
<:loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean

5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_meanVarHandleOp*
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *F
shared_name75MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean
ť
VMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Initializer/zeros*
dtype0*H
_class>
<:loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean

IMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean
ç
JMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_varianceVarHandleOp*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
dtype0*J
shared_name;9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
Ă
ZMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ŕ
DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm2MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DDMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

;MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Ť
1MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_10_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
TMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ţ
SMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
ŕ
UMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ě
^MobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

RMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul^MobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalUMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*'
_output_shapes
:*
T0
ö
NMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normalAddRMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/mulSMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*'
_output_shapes
:*
T0

1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsVarHandleOp*
shape:*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*B
shared_name31MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
_output_shapes
: 
ł
RMobilenetV1/Conv2d_11_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp1MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
_output_shapes
: 
˘
8MobilenetV1/Conv2d_11_depthwise/depthwise_weights/AssignAssignVariableOp1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsNMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0

EMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights
ż
DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

;MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

CMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ć
5MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVarsDMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ó
@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB*  ?*B
_class8
64loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
ű
/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gammaVarHandleOp*B
_class8
64loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
shape:
Ż
PMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ň
@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta
ř
.MobilenetV1/Conv2d_11_depthwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta
­
OMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ŕ
GMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB*    *H
_class>
<:loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:

5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
dtype0*F
shared_name75MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
ť
VMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Initializer/zeros*H
_class>
<:loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
dtype0

IMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*H
_class>
<:loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ç
JMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_varianceVarHandleOp*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance
Ă
ZMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*L
_classB
@>loc:@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance
ą
DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ć
SMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ă
DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm5MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

;MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Wě?
Ť
1MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_11_depthwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
JMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*
_output_shapes
:
Ę
IMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*
_output_shapes
: 
Ě
KMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*
_output_shapes
: 
Ż
TMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalJMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/shape*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*(
_output_shapes
:
á
HMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/mulMulTMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/TruncatedNormalKMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights
Ď
DMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normalAddHMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/mulIMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal/mean*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:*
T0
đ
'MobilenetV1/Conv2d_11_pointwise/weightsVarHandleOp*
shape:*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*8
shared_name)'MobilenetV1/Conv2d_11_pointwise/weights*
_output_shapes
: 

HMobilenetV1/Conv2d_11_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'MobilenetV1/Conv2d_11_pointwise/weights*
_output_shapes
: 
ú
.MobilenetV1/Conv2d_11_pointwise/weights/AssignAssignVariableOp'MobilenetV1/Conv2d_11_pointwise/weightsDMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0
é
;MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0

SMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *ŹĹ'8
Ő
cMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*(
_output_shapes
:
ä
TMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LosscMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
 
MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizerMulSMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/scaleTMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
ł
AMobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:*
dtype0
ą
2MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DConv2DMMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVarsAMobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
Ó
@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*B
_class8
64loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
dtype0
ű
/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gammaVarHandleOp*
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ň
@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_11_pointwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta
­
OMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:*A
_class7
53loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta
ŕ
GMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *H
_class>
<:loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean

5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *F
shared_name75MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean
ť
VMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Initializer/zeros*H
_class>
<:loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
dtype0

IMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*H
_class>
<:loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ç
JMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB*  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:

9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_varianceVarHandleOp*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
dtype0*J
shared_name;9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
Ă
ZMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ŕ
DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm2MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DDMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

;MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Ť
1MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_11_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
TMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0*
_output_shapes
:*%
valueB"            
Ţ
SMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
ŕ
UMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ěQ¸=*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0
Ě
^MobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights

RMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul^MobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalUMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:
ö
NMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normalAddRMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/mulSMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:

1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsVarHandleOp*
shape:*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0*
_output_shapes
: *B
shared_name31MobilenetV1/Conv2d_12_depthwise/depthwise_weights
ł
RMobilenetV1/Conv2d_12_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp1MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
_output_shapes
: 
˘
8MobilenetV1/Conv2d_12_depthwise/depthwise_weights/AssignAssignVariableOp1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsNMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0

EMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_12_depthwise/depthwise_weights*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
ż
DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

;MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0

CMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
Ć
5MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVarsDMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOp*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*B
_class8
64loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma
ű
/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gammaVarHandleOp*
dtype0*@
shared_name1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
_output_shapes
: *
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ň
@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_12_depthwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
dtype0*?
shared_name0.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
_output_shapes
: 
­
OMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ŕ
GMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *H
_class>
<:loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean

5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_meanVarHandleOp*H
_class>
<:loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *F
shared_name75MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
shape:
ť
VMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Initializer/zeros*H
_class>
<:loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
dtype0

IMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*H
_class>
<:loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ç
JMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Initializer/onesConst*L
_classB
@>loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:*
valueB*  ?

9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance
Ă
ZMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma*
_output_shapes	
:*
dtype0
˛
FMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ă
DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm5MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::*
T0

;MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ConstConst*
valueB
 *Wě?*
dtype0*
_output_shapes
: 
Ť
1MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_12_depthwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
JMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*
_output_shapes
:
Ę
IMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0
Ě
KMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*
_output_shapes
: 
Ż
TMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalJMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/shape*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*(
_output_shapes
:
á
HMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/mulMulTMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/TruncatedNormalKMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/stddev*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:*
T0
Ď
DMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normalAddHMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/mulIMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal/mean*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:
đ
'MobilenetV1/Conv2d_12_pointwise/weightsVarHandleOp*
shape:*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*
_output_shapes
: *8
shared_name)'MobilenetV1/Conv2d_12_pointwise/weights

HMobilenetV1/Conv2d_12_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'MobilenetV1/Conv2d_12_pointwise/weights*
_output_shapes
: 
ú
.MobilenetV1/Conv2d_12_pointwise/weights/AssignAssignVariableOp'MobilenetV1/Conv2d_12_pointwise/weightsDMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0
é
;MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0

SMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *ŹĹ'8*
dtype0
Ő
cMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*(
_output_shapes
:
ä
TMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LosscMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
 
MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizerMulSMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/scaleTMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ł
AMobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_12_pointwise/weights*
dtype0*(
_output_shapes
:
ą
2MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DConv2DMMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVarsAMobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
ß
PMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*
valueB:*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
dtype0
Ď
FMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma
Ě
@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/onesFillPMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorFMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma
ű
/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gammaVarHandleOp*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
shape:
Ż
PMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones*
dtype0*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma
ô
CMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ţ
PMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
:
Î
FMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros/ConstConst*
valueB
 *    *A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ë
@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zerosFillPMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorFMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros/Const*
T0*A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_12_pointwise/BatchNorm/betaVarHandleOp*A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
shape:
­
OMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:*A
_class7
53loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta
ě
WMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
dtype0
Ü
MMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ç
GMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zerosFillWMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorMMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros/Const*H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
_output_shapes	
:*
T0

5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_meanVarHandleOp*H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *F
shared_name75MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
shape:
ť
VMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros*
dtype0*H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean

IMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
_output_shapes	
:*H
_class>
<:loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
dtype0
ó
ZMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:
ă
PMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
ô
JMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/onesFillZMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorPMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance

9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *J
shared_name;9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0
Ă
ZMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ŕ
DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm2MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DDMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

;MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *Wě?
Ť
1MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_12_pointwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
TMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*
_output_shapes
:
Ţ
SMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
ŕ
UMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
Ě
^MobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/shape*'
_output_shapes
:*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0

RMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/mulMul^MobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalUMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*'
_output_shapes
:
ö
NMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normalAddRMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/mulSMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*'
_output_shapes
:*
T0

1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsVarHandleOp*
shape:*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*B
shared_name31MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
_output_shapes
: 
ł
RMobilenetV1/Conv2d_13_depthwise/depthwise_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp1MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
_output_shapes
: 
˘
8MobilenetV1/Conv2d_13_depthwise/depthwise_weights/AssignAssignVariableOp1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsNMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0

EMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_13_depthwise/depthwise_weights*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:
ż
DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOpReadVariableOp1MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:

;MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

CMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ć
5MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVarsDMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOp*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
ß
PMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
:
Ď
FMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: 
Ě
@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/onesFillPMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorFMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones/Const*
T0*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
_output_shapes	
:
ű
/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gammaVarHandleOp*
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ţ
PMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
:
Î
FMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros/ConstConst*
valueB
 *    *A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ë
@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zerosFillPMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorFMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros/Const*A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
_output_shapes	
:*
T0
ř
.MobilenetV1/Conv2d_13_depthwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta
­
OMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ě
WMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:
Ü
MMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ç
GMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zerosFillWMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorMMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
_output_shapes	
:

5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_meanVarHandleOp*
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *F
shared_name75MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean
ť
VMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros*H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
dtype0

IMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*H
_class>
<:loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ó
ZMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:
ă
PMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
dtype0
ô
JMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/onesFillZMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorPMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance

9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_varianceVarHandleOp*
dtype0*J
shared_name;9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance
Ă
ZMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
_output_shapes	
:*L
_classB
@>loc:@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
dtype0
ą
DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta*
_output_shapes	
:*
dtype0
Ć
SMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean*
_output_shapes	
:*
dtype0
Ě
UMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ă
DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormFusedBatchNorm5MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

;MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ConstConst*
_output_shapes
: *
valueB
 *Wě?*
dtype0
Ť
1MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_13_depthwise/add_fold*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
JMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/shapeConst*%
valueB"            *:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*
_output_shapes
:
Ę
IMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*
_output_shapes
: 
Ě
KMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/stddevConst*
valueB
 *ěQ¸=*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*
_output_shapes
: 
Ż
TMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalJMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/shape*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*(
_output_shapes
:
á
HMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/mulMulTMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/TruncatedNormalKMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights
Ď
DMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normalAddHMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/mulIMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal/mean*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*(
_output_shapes
:
đ
'MobilenetV1/Conv2d_13_pointwise/weightsVarHandleOp*
shape:*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*8
shared_name)'MobilenetV1/Conv2d_13_pointwise/weights*
_output_shapes
: 

HMobilenetV1/Conv2d_13_pointwise/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'MobilenetV1/Conv2d_13_pointwise/weights*
_output_shapes
: 
ú
.MobilenetV1/Conv2d_13_pointwise/weights/AssignAssignVariableOp'MobilenetV1/Conv2d_13_pointwise/weightsDMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0
é
;MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_13_pointwise/weights*(
_output_shapes
:*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
dtype0

SMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *ŹĹ'8
Ő
cMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*(
_output_shapes
:
ä
TMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2LossL2LosscMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
_output_shapes
: *
T0
 
MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizerMulSMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/scaleTMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0

9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ł
AMobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOpReadVariableOp'MobilenetV1/Conv2d_13_pointwise/weights*
dtype0*(
_output_shapes
:
ą
2MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DConv2DMMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVarsAMobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
PMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma
Ď
FMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
dtype0
Ě
@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/onesFillPMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorFMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones/Const*
T0*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
_output_shapes	
:
ű
/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gammaVarHandleOp*
shape:*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes
: *@
shared_name1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma
Ż
PMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
_output_shapes
: 

6MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/AssignAssignVariableOp/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
dtype0
ô
CMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*B
_class8
64loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
Ţ
PMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta
Î
FMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros/ConstConst*
valueB
 *    *A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
dtype0*
_output_shapes
: 
Ë
@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zerosFillPMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorFMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros/Const*
T0*A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
_output_shapes	
:
ř
.MobilenetV1/Conv2d_13_pointwise/BatchNorm/betaVarHandleOp*
shape:*A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
dtype0*?
shared_name0.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
_output_shapes
: 
­
OMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
_output_shapes
: 

5MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/AssignAssignVariableOp.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros*A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
dtype0
ń
BMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOpReadVariableOp.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*A
_class7
53loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
ě
WMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
:
Ü
MMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
ç
GMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zerosFillWMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorMMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros/Const*H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
_output_shapes	
:*
T0

5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_meanVarHandleOp*
dtype0*F
shared_name75MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
_output_shapes
: *
shape:*H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean
ť
VMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
_output_shapes
: 
§
<MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/AssignAssignVariableOp5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_meanGMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros*H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
dtype0

IMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*H
_class>
<:loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
ó
ZMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
:
ă
PMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
ô
JMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/onesFillZMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorPMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones/Const*
T0*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
_output_shapes	
:

9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_varianceVarHandleOp*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0*J
shared_name;9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
_output_shapes
: *
shape:
Ă
ZMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
_output_shapes
: 
ś
@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/AssignAssignVariableOp9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_varianceJMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0

MMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*L
_classB
@>loc:@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
ą
DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOpReadVariableOp/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma*
dtype0*
_output_shapes	
:
˛
FMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1ReadVariableOp.MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta*
dtype0*
_output_shapes	
:
Ć
SMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpReadVariableOp5MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean*
dtype0*
_output_shapes	
:
Ě
UMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1ReadVariableOp9MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance*
dtype0*
_output_shapes	
:
Ŕ
DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormFusedBatchNorm2MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DDMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOpFMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1SMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOpUMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1*
T0*
is_training( *
epsilon%o:*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙::::

;MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ConstConst*
_output_shapes
: *
valueB
 *Wě?*
dtype0
Ť
1MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6Relu64MobilenetV1/MobilenetV1/Conv2d_13_pointwise/add_fold*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
%MobilenetV1/Logits/AvgPool_1a/AvgPoolAvgPoolMMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingVALID*
ksize
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
(hub_output/feature_vector/SpatialSqueezeSqueeze%MobilenetV1/Logits/AvgPool_1a/AvgPool*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
squeeze_dims

Ą
8hub_output/feature_vector/SpatialSqueeze/ReadForQuantizeIdentity(hub_output/feature_vector/SpatialSqueeze*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
5MobilenetV1/Logits/AvgPool_1a/AvgPool/ReadForQuantizeIdentity%MobilenetV1/Logits/AvgPool_1a/AvgPool*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 
Ŕ
6MobilenetV1/MobilenetV1/Conv2d_0/Relu6/ReadForQuantizeIdentityBMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp *
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
Ő
@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
AMobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
AMobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
AMobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
AMobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
AMobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
Ő
@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@*
T0
Ő
@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6/ReadForQuantizeIdentityLMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
AMobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
AMobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
AMobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6/ReadForQuantizeIdentityMMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVars*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
5MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
â
3MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/addAddJMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOp_15MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/add/y*
_output_shapes
: *
T0

5MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/RsqrtRsqrt3MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/add*
_output_shapes
: *
T0
Ń
3MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/mulMul9MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp5MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes
: 
ŕ
5MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/mul_1MulHMobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm/ReadVariableOp3MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/mul*
_output_shapes
: *
T0
Ô
4MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/biasSub;MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/ReadVariableOp_15MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/mul_1*
_output_shapes
: *
T0
Î
)MobilenetV1/MobilenetV1/Conv2d_0/mul_foldMul6MobilenetV1/MobilenetV1/Conv2d_0/Conv2D/ReadVariableOp3MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/mul*
T0*&
_output_shapes
: 

,MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_FoldConv2D#hub_input/FakeQuantWithMinMaxArgs_1FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars*
strides
*
T0*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 
Î
)MobilenetV1/MobilenetV1/Conv2d_0/add_foldAdd,MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_Fold4MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/bias*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 

?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/add/y*
_output_shapes
: *
T0
Ź
?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/add*
_output_shapes
: *
T0
ď
=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes
: 
ţ
?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes
: 
ň
>MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/mul_1*
_output_shapes
: *
T0

MMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"       

GMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes

: *
T0
ů
3MobilenetV1/MobilenetV1/Conv2d_1_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/scale_reshape*
T0*&
_output_shapes
: 

?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold/ShapeConst*
_output_shapes
:*%
valueB"             *
dtype0

GMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ę
9MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_FoldDepthwiseConv2dNativeBMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp *
T0*
strides
*
paddingSAME
ď
3MobilenetV1/MobilenetV1/Conv2d_1_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm_Fold/bias*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 

?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/add/y*
_output_shapes
:@*
T0
Ź
?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/add*
_output_shapes
:@*
T0
ď
=MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes
:@*
T0
ţ
?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes
:@
ň
>MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/mul_1*
_output_shapes
:@*
T0
ě
3MobilenetV1/MobilenetV1/Conv2d_1_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/mul*
T0*&
_output_shapes
: @
Â
6MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@*
strides
*
T0*
paddingSAME
ě
3MobilenetV1/MobilenetV1/Conv2d_1_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm_Fold/bias*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@

?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/add/y*
_output_shapes
:@*
T0
Ź
?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/add*
T0*
_output_shapes
:@
ď
=MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes
:@*
T0
ţ
?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes
:@
ň
>MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/mul_1*
_output_shapes
:@*
T0

MMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0

GMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes

:@*
T0
ů
3MobilenetV1/MobilenetV1/Conv2d_2_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/scale_reshape*
T0*&
_output_shapes
:@

?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      

GMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ô
9MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@*
T0*
strides
*
paddingSAME
ď
3MobilenetV1/MobilenetV1/Conv2d_2_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm_Fold/bias*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@

?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
­
?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
í
3MobilenetV1/MobilenetV1/Conv2d_2_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/mul*'
_output_shapes
:@*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
í
3MobilenetV1/MobilenetV1/Conv2d_2_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88

?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

MMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/scale_reshape/shape*
T0*
_output_shapes
:	
ú
3MobilenetV1/MobilenetV1/Conv2d_3_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

GMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_Fold/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ő
9MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0*
strides
*
paddingSAME
đ
3MobilenetV1/MobilenetV1/Conv2d_3_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88*
T0

?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/add*
T0*
_output_shapes	
:
đ
=MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
î
3MobilenetV1/MobilenetV1/Conv2d_3_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/mul*
T0*(
_output_shapes
:
Ă
6MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars*
strides
*
T0*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
í
3MobilenetV1/MobilenetV1/Conv2d_3_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88

?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:

MMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      

GMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/scale_reshape/shape*
T0*
_output_shapes
:	
ú
3MobilenetV1/MobilenetV1/Conv2d_4_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_Fold/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
Ő
9MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars*
strides
*
T0*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
3MobilenetV1/MobilenetV1/Conv2d_4_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
î
3MobilenetV1/MobilenetV1/Conv2d_4_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
í
3MobilenetV1/MobilenetV1/Conv2d_4_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

MMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ú
3MobilenetV1/MobilenetV1/Conv2d_5_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/scale_reshape*'
_output_shapes
:*
T0

?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ő
9MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
đ
3MobilenetV1/MobilenetV1/Conv2d_5_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0
î
3MobilenetV1/MobilenetV1/Conv2d_5_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
T0*
paddingSAME
í
3MobilenetV1/MobilenetV1/Conv2d_5_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
­
?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:

MMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0

GMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ú
3MobilenetV1/MobilenetV1/Conv2d_6_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/scale_reshape*'
_output_shapes
:*
T0

?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

GMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ő
9MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
3MobilenetV1/MobilenetV1/Conv2d_6_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
­
?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/add*
T0*
_output_shapes	
:
đ
=MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
î
3MobilenetV1/MobilenetV1/Conv2d_6_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/mul*
T0*(
_output_shapes
:
Ă
6MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
T0*
paddingSAME
í
3MobilenetV1/MobilenetV1/Conv2d_6_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/add*
T0*
_output_shapes	
:
đ
=MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

MMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ú
3MobilenetV1/MobilenetV1/Conv2d_7_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

GMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ő
9MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides

đ
3MobilenetV1/MobilenetV1/Conv2d_7_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/add*
T0*
_output_shapes	
:
đ
=MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0
î
3MobilenetV1/MobilenetV1/Conv2d_7_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
3MobilenetV1/MobilenetV1/Conv2d_7_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

MMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/scale_reshape/shape*
T0*
_output_shapes
:	
ú
3MobilenetV1/MobilenetV1/Conv2d_8_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/scale_reshape*'
_output_shapes
:*
T0

?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ő
9MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
đ
3MobilenetV1/MobilenetV1/Conv2d_8_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
­
?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0
î
3MobilenetV1/MobilenetV1/Conv2d_8_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
3MobilenetV1/MobilenetV1/Conv2d_8_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
­
?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0
˙
?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:

MMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/scale_reshapeReshape=MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/mulMMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ú
3MobilenetV1/MobilenetV1/Conv2d_9_depthwise/mul_foldMulCMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ReadVariableOpGMobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/scale_reshape*'
_output_shapes
:*
T0

?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

GMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ő
9MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
3MobilenetV1/MobilenetV1/Conv2d_9_depthwise/add_foldAdd9MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_Fold>MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/addAddTMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
­
?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/RsqrtRsqrt=MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
đ
=MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/mulMulCMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/Rsqrt*
T0*
_output_shapes	
:
˙
?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/mul_1MulRMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/biasSubEMobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/ReadVariableOp_1?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
î
3MobilenetV1/MobilenetV1/Conv2d_9_pointwise/mul_foldMul@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D/ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ă
6MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_FoldConv2DLMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVarsPMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
strides
*
T0*
paddingSAME
í
3MobilenetV1/MobilenetV1/Conv2d_9_pointwise/add_foldAdd6MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_Fold>MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

>MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ö
?MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

NMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

HMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/scale_reshapeReshape>MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/mulNMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ý
4MobilenetV1/MobilenetV1/Conv2d_10_depthwise/mul_foldMulDMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ReadVariableOpHMobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

HMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
×
:MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_FoldDepthwiseConv2dNativeLMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
ó
4MobilenetV1/MobilenetV1/Conv2d_10_depthwise/add_foldAdd:MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_Fold?MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

>MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ö
?MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0
ń
4MobilenetV1/MobilenetV1/Conv2d_10_pointwise/mul_foldMulAMobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/mul*
T0*(
_output_shapes
:
Ć
7MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_FoldConv2DMMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
đ
4MobilenetV1/MobilenetV1/Conv2d_10_pointwise/add_foldAdd7MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_Fold?MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

>MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ö
?MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:

NMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      

HMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/scale_reshapeReshape>MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/mulNMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/scale_reshape/shape*
_output_shapes
:	*
T0
ý
4MobilenetV1/MobilenetV1/Conv2d_11_depthwise/mul_foldMulDMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ReadVariableOpHMobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

HMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_Fold/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ř
:MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_FoldDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides

ó
4MobilenetV1/MobilenetV1/Conv2d_11_depthwise/add_foldAdd:MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_Fold?MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

>MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
Ż
@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ö
?MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
ń
4MobilenetV1/MobilenetV1/Conv2d_11_pointwise/mul_foldMulAMobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ć
7MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_FoldConv2DMMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars*
strides
*
T0*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
4MobilenetV1/MobilenetV1/Conv2d_11_pointwise/add_foldAdd7MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_Fold?MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:

>MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/add/y*
T0*
_output_shapes	
:
Ż
@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/mul*
T0*
_output_shapes	
:
ö
?MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

NMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

HMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/scale_reshapeReshape>MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/mulNMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/scale_reshape/shape*
T0*
_output_shapes
:	
ý
4MobilenetV1/MobilenetV1/Conv2d_12_depthwise/mul_foldMulDMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ReadVariableOpHMobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/scale_reshape*
T0*'
_output_shapes
:

@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_Fold/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:

HMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ř
:MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_FoldDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars*
T0*
strides
*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
4MobilenetV1/MobilenetV1/Conv2d_12_depthwise/add_foldAdd:MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_Fold?MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

>MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ö
?MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0
ń
4MobilenetV1/MobilenetV1/Conv2d_12_pointwise/mul_foldMulAMobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ć
7MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D_FoldConv2DMMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
đ
4MobilenetV1/MobilenetV1/Conv2d_12_pointwise/add_foldAdd7MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D_Fold?MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

>MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/add*
T0*
_output_shapes	
:
ó
>MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ö
?MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/mul_1*
_output_shapes	
:*
T0

NMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/scale_reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

HMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/scale_reshapeReshape>MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/mulNMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/scale_reshape/shape*
T0*
_output_shapes
:	
ý
4MobilenetV1/MobilenetV1/Conv2d_13_depthwise/mul_foldMulDMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ReadVariableOpHMobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/scale_reshape*'
_output_shapes
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_Fold/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            

HMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_Fold/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ř
:MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_FoldDepthwiseConv2dNativeMMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
strides
*
paddingSAME
ó
4MobilenetV1/MobilenetV1/Conv2d_13_depthwise/add_foldAdd:MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_Fold?MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm_Fold/bias*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0

>MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/addAddUMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/add/y*
_output_shapes	
:*
T0
Ż
@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/RsqrtRsqrt>MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/add*
_output_shapes	
:*
T0
ó
>MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/mulMulDMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/Rsqrt*
_output_shapes	
:*
T0

@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/mul_1MulSMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/mul*
_output_shapes	
:*
T0
ö
?MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/biasSubFMobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/ReadVariableOp_1@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/mul_1*
T0*
_output_shapes	
:
ń
4MobilenetV1/MobilenetV1/Conv2d_13_pointwise/mul_foldMulAMobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D/ReadVariableOp>MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/mul*(
_output_shapes
:*
T0
Ć
7MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D_FoldConv2DMMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVarsQMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars*
strides
*
T0*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
đ
4MobilenetV1/MobilenetV1/Conv2d_13_pointwise/add_foldAdd7MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D_Fold?MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm_Fold/bias*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
DMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  ŔŔ*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
dtype0
˙
2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/minVarHandleOp*
shape: *E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
dtype0*
_output_shapes
: *C
shared_name42MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min
ľ
SMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/AssignAssignVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/minDMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/Initializer/Const*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
dtype0
ř
FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/Read/ReadVariableOpReadVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
dtype0*
_output_shapes
: 
Đ
DMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*
dtype0*
_output_shapes
: 
˙
2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/maxVarHandleOp*
dtype0*C
shared_name42MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*
_output_shapes
: *
shape: *E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max
ľ
SMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/AssignAssignVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/maxDMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/Initializer/Const*
dtype0*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max
ř
FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/Read/ReadVariableOpReadVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*E
_class;
97loc:@MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*
dtype0*
_output_shapes
: 
Ŕ
UMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min*
dtype0*
_output_shapes
: 
Â
WMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp2MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max*
dtype0*
_output_shapes
: 
ř
FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars)MobilenetV1/MobilenetV1/Conv2d_0/mul_foldUMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpWMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*&
_output_shapes
: *
narrow_range(
Č
@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/Initializer/ConstConst*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
dtype0*
_output_shapes
: *
valueB
 *    
ó
.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/minVarHandleOp*
shape: *A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min
­
OMobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
_output_shapes
: 

5MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/AssignAssignVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/Initializer/Const*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
dtype0
ě
BMobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/Read/ReadVariableOpReadVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
dtype0*
_output_shapes
: 
Č
@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
dtype0
ó
.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/maxVarHandleOp*
shape: *A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
dtype0*
_output_shapes
: *?
shared_name0.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max
­
OMobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
_output_shapes
: 

5MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/AssignAssignVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/Initializer/Const*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
dtype0
ě
BMobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/Read/ReadVariableOpReadVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*A
_class7
53loc:@MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
dtype0*
_output_shapes
: 
¸
QMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min*
dtype0*
_output_shapes
: 
ş
SMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp.MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max*
dtype0*
_output_shapes
: 
Ţ
BMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars&MobilenetV1/MobilenetV1/Conv2d_0/Relu6QMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpSMobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 
ä
NMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/Initializer/Const*
dtype0*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min

PMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min
ä
NMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/maxVarHandleOp*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
shape: 
É
]MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/Initializer/Const*
dtype0*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max

PMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
PMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_1_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*&
_output_shapes
: *
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/Initializer/ConstConst*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
dtype0*
_output_shapes
: *
valueB
 *    

8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min
Ü
JMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp 
ä
NMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
PMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_1_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*&
_output_shapes
: @*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min*
_output_shapes
: *
dtype0
Î
]MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙pp@
ä
NMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/minVarHandleOp*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
dtype0
É
]MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/maxVarHandleOp*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
shape: 
É
]MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
PMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_2_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*
narrow_range(*&
_output_shapes
:@
Ü
JMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/minVarHandleOp*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
_output_shapes
: *
shape: 
Á
YMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
dtype0

8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/maxVarHandleOp*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
_output_shapes
: *
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙88@
ä
NMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/minVarHandleOp*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
dtype0
É
]MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max

<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
dtype0
Ô
_MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_2_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:@*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/minVarHandleOp*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
_output_shapes
: *
shape: 
Á
YMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
dtype0

8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max
Ě
[MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
ä
NMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
dtype0
ä
NMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_3_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
dtype0
Ü
JMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max

8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/maxVarHandleOp*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
dtype0
Á
YMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
ä
NMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/minVarHandleOp*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
_output_shapes
: *
shape: 
É
]MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
dtype0*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min
ä
NMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max

<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min*
_output_shapes
: *
dtype0
Ö
aMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_3_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙88
ä
NMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/Initializer/Const*
dtype0*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max

PMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max*
_output_shapes
: *
dtype0
Ą
PMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_4_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/Initializer/ConstConst*
_output_shapes
: *
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
dtype0

8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/maxVarHandleOp*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
_output_shapes
: *
shape: 
Á
YMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/Initializer/ConstConst*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
dtype0*
_output_shapes
: *
valueB
 *  ŔŔ

<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_4_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
dtype0

8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
dtype0
Ô
_MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_5_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
dtype0
Ü
JMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max

8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/maxVarHandleOp*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
dtype0
Á
YMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/minVarHandleOp*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
_output_shapes
: *
shape: 
É
]MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max

<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_5_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/Initializer/ConstConst*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *
valueB
 *  ŔŔ

<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/minVarHandleOp*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max
Ô
_MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_6_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*
narrow_range(*'
_output_shapes
:
Ü
JMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/minVarHandleOp*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
shape: 
Á
YMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min
Ü
JMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max
Ě
[MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
dtype0
ä
NMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/maxVarHandleOp*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
dtype0
É
]MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_6_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max

8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max
Ě
[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min

<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_7_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/maxVarHandleOp*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
_output_shapes
: *
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
dtype0
Á
YMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max

LMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/minVarHandleOp*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
_output_shapes
: *
shape: 
É
]MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/maxVarHandleOp*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
dtype0
É
]MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max
Ô
_MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_7_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/Initializer/ConstConst*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@

8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
dtype0
Ě
[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
dtype0
ä
NMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ą
PMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_8_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min

8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/Initializer/Const*
dtype0*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min

LMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
dtype0

<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_8_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max
Á
YMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/Initializer/ConstConst*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *
valueB
 *  ŔŔ

<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max
É
]MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ô
_MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max*
_output_shapes
: *
dtype0
Ą
PMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_9_depthwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min
Ü
JMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/maxVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
dtype0*I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
_output_shapes
: 
Á
YMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
dtype0*
_output_shapes
: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max
Ě
[MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
NMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/minVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
dtype0*
_output_shapes
: *M
shared_name><MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min
É
]MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/minNMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ä
NMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 

<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/maxVarHandleOp*
shape: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
dtype0*M
shared_name><MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
_output_shapes
: 
É
]MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
_output_shapes
: 
Ă
CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/AssignAssignVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/maxNMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/Initializer/Const*O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
dtype0

PMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
_output_shapes
: *O
_classE
CAloc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
dtype0
Ô
_MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ö
aMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp<MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
˘
PMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars3MobilenetV1/MobilenetV1/Conv2d_9_pointwise/mul_fold_MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpaMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ü
JMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/minVarHandleOp*
shape: *K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min
Á
YMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/minJMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ü
JMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/Initializer/ConstConst*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@

8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/maxVarHandleOp*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
dtype0*
_output_shapes
: *I
shared_name:8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
shape: 
Á
YMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
_output_shapes
: 
ł
?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/AssignAssignVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/maxJMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/Initializer/Const*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
dtype0

LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*K
_classA
?=loc:@MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Ě
[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Î
]MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6[MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp]MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
_output_shapes
: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
dtype0
ć
OMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max
Ë
^MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/Initializer/Const*
dtype0*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max

QMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ľ
QMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_10_depthwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*
narrow_range(*'
_output_shapes
:
Ţ
KMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min

9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/minVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
dtype0*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
_output_shapes
: 
Ă
ZMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
_output_shapes
: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
dtype0
Ţ
KMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
dtype0*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
_output_shapes
: 
Ă
ZMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ć
OMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/maxVarHandleOp*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
shape: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ś
QMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_10_pointwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*
narrow_range(*(
_output_shapes
:
Ţ
KMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/Initializer/ConstConst*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
dtype0*
_output_shapes
: *
valueB
 *    

9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/minVarHandleOp*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
_output_shapes
: *
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
dtype0
Ă
ZMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ţ
KMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max
Ă
ZMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/Initializer/Const*
dtype0*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max

MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/minVarHandleOp*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min
Ë
^MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
ć
OMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max
 
=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/maxVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ľ
QMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_11_depthwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/minVarHandleOp*
dtype0*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
_output_shapes
: *
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min
Ă
ZMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ţ
KMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max
Ă
ZMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/minVarHandleOp*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
shape: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ć
OMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
dtype0
 
=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/maxVarHandleOp*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
shape: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min*
_output_shapes
: *
dtype0
Ř
bMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ś
QMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_11_pointwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/Initializer/ConstConst*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
dtype0*
_output_shapes
: *
valueB
 *    

9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/minVarHandleOp*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
dtype0
Ă
ZMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/Initializer/Const*
dtype0*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min

MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
dtype0*
_output_shapes
: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min
Ţ
KMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/maxVarHandleOp*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max
Ă
ZMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
_output_shapes
: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
dtype0
Î
\MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min
 
=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min
Ë
^MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min*
dtype0*
_output_shapes
: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min
ć
OMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/maxVarHandleOp*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
_output_shapes
: *
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
dtype0
Ë
^MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ľ
QMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_12_depthwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/minVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
dtype0*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
_output_shapes
: 
Ă
ZMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/Initializer/Const*
dtype0*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min

MMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Ţ
KMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/Initializer/ConstConst*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@

9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
dtype0*J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
_output_shapes
: 
Ă
ZMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
_output_shapes
: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
dtype0
ć
OMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/maxVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max
Ë
^MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/Initializer/Const*
dtype0*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max

QMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max*
_output_shapes
: *
dtype0
Ś
QMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_12_pointwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min

9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/minVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min
Ă
ZMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ţ
KMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max
Ă
ZMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min*
_output_shapes
: *
dtype0
Đ
^MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max*
_output_shapes
: *
dtype0

MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/Initializer/ConstConst*
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
 
=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
_output_shapes
: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
dtype0
ć
OMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max
 
=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/maxVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ľ
QMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_13_depthwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*'
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/minVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min
Ă
ZMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
dtype0*
_output_shapes
: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min
Ţ
KMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/Initializer/ConstConst*
valueB
 *  Ŕ@*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/maxVarHandleOp*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
shape: 
Ă
ZMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max*
dtype0*
_output_shapes
: 

MMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
OMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/Initializer/ConstConst*
_output_shapes
: *
valueB
 *  ŔŔ*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
dtype0
 
=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/minVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
dtype0*
_output_shapes
: *N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min
Ë
^MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/minOMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
ć
OMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/Initializer/ConstConst*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@
 
=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/maxVarHandleOp*
shape: *P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
dtype0*N
shared_name?=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
_output_shapes
: 
Ë
^MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
_output_shapes
: 
Ç
DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/AssignAssignVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/maxOMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/Initializer/Const*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
dtype0

QMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/Read/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*P
_classF
DBloc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ö
`MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min*
dtype0*
_output_shapes
: 
Ř
bMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max*
dtype0*
_output_shapes
: 
Ś
QMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4MobilenetV1/MobilenetV1/Conv2d_13_pointwise/mul_fold`MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOpbMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*(
_output_shapes
:*
narrow_range(
Ţ
KMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/Initializer/ConstConst*
valueB
 *    *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
dtype0*
_output_shapes
: 

9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/minVarHandleOp*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
dtype0
Ă
ZMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/minKMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Ţ
KMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/Initializer/ConstConst*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
dtype0*
_output_shapes
: *
valueB
 *  Ŕ@

9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/maxVarHandleOp*
shape: *L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
dtype0*
_output_shapes
: *J
shared_name;9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max
Ă
ZMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/IsInitialized/VarIsInitializedOpVarIsInitializedOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
_output_shapes
: 
ˇ
@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/AssignAssignVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/maxKMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/Initializer/Const*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
dtype0

MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/Read/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*L
_classB
@>loc:@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
dtype0*
_output_shapes
: 
Î
\MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min*
dtype0*
_output_shapes
: 
Đ
^MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max*
_output_shapes
: *
dtype0

MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6\MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp^MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars/ReadVariableOp_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙"";
hub_module_attachments!

image_module_info
ŕŕ"÷
regularization_lossesÝ
Ú
DMobilenetV1/MobilenetV1/Conv2d_0/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_1_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_2_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_3_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_4_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_5_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_6_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_7_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_8_pointwise/kernel/Regularizer/l2_regularizer:0
NMobilenetV1/MobilenetV1/Conv2d_9_pointwise/kernel/Regularizer/l2_regularizer:0
OMobilenetV1/MobilenetV1/Conv2d_10_pointwise/kernel/Regularizer/l2_regularizer:0
OMobilenetV1/MobilenetV1/Conv2d_11_pointwise/kernel/Regularizer/l2_regularizer:0
OMobilenetV1/MobilenetV1/Conv2d_12_pointwise/kernel/Regularizer/l2_regularizer:0
OMobilenetV1/MobilenetV1/Conv2d_13_pointwise/kernel/Regularizer/l2_regularizer:0"é
trainable_variablesĐĚ
ş
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign2MobilenetV1/Conv2d_0/weights/Read/ReadVariableOp:0(2;MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal:08
Î
&MobilenetV1/Conv2d_0/BatchNorm/gamma:0+MobilenetV1/Conv2d_0/BatchNorm/gamma/Assign:MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/gamma/Initializer/ones:08
Ë
%MobilenetV1/Conv2d_0/BatchNorm/beta:0*MobilenetV1/Conv2d_0/BatchNorm/beta/Assign9MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign<MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign<MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign<MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign<MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign<MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign<MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign<MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign<MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Initializer/zeros:08

2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Initializer/zeros:08
â
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign<MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Initializer/zeros:08

3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Initializer/zeros:08
ć
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign=MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Initializer/zeros:08

3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Initializer/zeros:08
ć
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign=MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Initializer/zeros:08

3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Initializer/zeros:08
ć
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign=MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros:08

3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros:08
ć
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign=MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros:08"Ü
	variablesÍÉ
ş
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign2MobilenetV1/Conv2d_0/weights/Read/ReadVariableOp:0(2;MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal:08
Î
&MobilenetV1/Conv2d_0/BatchNorm/gamma:0+MobilenetV1/Conv2d_0/BatchNorm/gamma/Assign:MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/gamma/Initializer/ones:08
Ë
%MobilenetV1/Conv2d_0/BatchNorm/beta:0*MobilenetV1/Conv2d_0/BatchNorm/beta/Assign9MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/beta/Initializer/zeros:08
é
,MobilenetV1/Conv2d_0/BatchNorm/moving_mean:01MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Assign@MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Read/ReadVariableOp:0(2>MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Initializer/zeros:0@H
ř
0MobilenetV1/Conv2d_0/BatchNorm/moving_variance:05MobilenetV1/Conv2d_0/BatchNorm/moving_variance/AssignDMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign<MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign<MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign<MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign<MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign<MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign<MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign<MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign<MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign<MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign=MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign=MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign=MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign=MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

4MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min:09MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/AssignHMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/Read/ReadVariableOp:0(2FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/min/Initializer/Const:0

4MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max:09MobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/AssignHMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/Read/ReadVariableOp:0(2FMobilenetV1/MobilenetV1/Conv2d_0/weights_quant/max/Initializer/Const:0
ő
0MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min:05MobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/AssignDMobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/Read/ReadVariableOp:0(2BMobilenetV1/MobilenetV1/Conv2d_0/act_quant/min/Initializer/Const:0
ő
0MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max:05MobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/AssignDMobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/Read/ReadVariableOp:0(2BMobilenetV1/MobilenetV1/Conv2d_0/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_1_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_1_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_1_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_1_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_2_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_2_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_2_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_2_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_3_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_3_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_3_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_3_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_4_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_4_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_4_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_4_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_5_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_5_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_5_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_5_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_6_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_6_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_6_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_6_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_7_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_7_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_7_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_7_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_8_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_8_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_8_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_8_pointwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_9_depthwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_9_depthwise/act_quant/max/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min:0CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/AssignRMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/min/Initializer/Const:0
­
>MobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max:0CMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/AssignRMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/Read/ReadVariableOp:0(2PMobilenetV1/MobilenetV1/Conv2d_9_pointwise/weights_quant/max/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min:0?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/AssignNMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/min/Initializer/Const:0

:MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max:0?MobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/AssignNMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/Read/ReadVariableOp:0(2LMobilenetV1/MobilenetV1/Conv2d_9_pointwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_10_depthwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_10_depthwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_10_pointwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_10_pointwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_11_depthwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_11_depthwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_11_pointwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_11_pointwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_12_depthwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_12_depthwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_12_pointwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_12_pointwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_13_depthwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_13_depthwise/act_quant/max/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min:0DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/AssignSMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/min/Initializer/Const:0
ą
?MobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max:0DMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/AssignSMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/Read/ReadVariableOp:0(2QMobilenetV1/MobilenetV1/Conv2d_13_pointwise/weights_quant/max/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min:0@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/AssignOMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/min/Initializer/Const:0
Ą
;MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max:0@MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/AssignOMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/Read/ReadVariableOp:0(2MMobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/max/Initializer/Const:0"˘
model_variables
ş
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign2MobilenetV1/Conv2d_0/weights/Read/ReadVariableOp:0(2;MobilenetV1/Conv2d_0/weights/Initializer/truncated_normal:08
Î
&MobilenetV1/Conv2d_0/BatchNorm/gamma:0+MobilenetV1/Conv2d_0/BatchNorm/gamma/Assign:MobilenetV1/Conv2d_0/BatchNorm/gamma/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/gamma/Initializer/ones:08
Ë
%MobilenetV1/Conv2d_0/BatchNorm/beta:0*MobilenetV1/Conv2d_0/BatchNorm/beta/Assign9MobilenetV1/Conv2d_0/BatchNorm/beta/Read/ReadVariableOp:0(27MobilenetV1/Conv2d_0/BatchNorm/beta/Initializer/zeros:08
é
,MobilenetV1/Conv2d_0/BatchNorm/moving_mean:01MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Assign@MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Read/ReadVariableOp:0(2>MobilenetV1/Conv2d_0/BatchNorm/moving_mean/Initializer/zeros:0@H
ř
0MobilenetV1/Conv2d_0/BatchNorm/moving_variance:05MobilenetV1/Conv2d_0/BatchNorm/moving_variance/AssignDMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_0/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign<MobilenetV1/Conv2d_1_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_1_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign<MobilenetV1/Conv2d_2_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_2_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign<MobilenetV1/Conv2d_3_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_3_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign<MobilenetV1/Conv2d_4_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_4_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign<MobilenetV1/Conv2d_5_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_5_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign<MobilenetV1/Conv2d_6_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_6_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign<MobilenetV1/Conv2d_7_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_7_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign<MobilenetV1/Conv2d_8_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_8_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/AssignFMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Read/ReadVariableOp:0(2OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_depthwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
â
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign<MobilenetV1/Conv2d_9_pointwise/weights/Read/ReadVariableOp:0(2EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/truncated_normal:08
ö
0MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma:05MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/AssignDMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma/Initializer/ones:08
ó
/MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta:04MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/AssignCMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2AMobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/Initializer/zeros:08

6MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean:0;MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/AssignJMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2HMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
 
:MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance:0?MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/AssignNMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2KMobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign=MobilenetV1/Conv2d_10_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_10_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign=MobilenetV1/Conv2d_11_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_11_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign=MobilenetV1/Conv2d_12_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_12_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H

3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/AssignGMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Read/ReadVariableOp:0(2PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_depthwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance/Initializer/ones:0@H
ć
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign=MobilenetV1/Conv2d_13_pointwise/weights/Read/ReadVariableOp:0(2FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/truncated_normal:08
ú
1MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:06MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/AssignEMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma/Initializer/ones:08
÷
0MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:05MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/AssignDMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Read/ReadVariableOp:0(2BMobilenetV1/Conv2d_13_pointwise/BatchNorm/beta/Initializer/zeros:08

7MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean:0<MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/AssignKMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Read/ReadVariableOp:0(2IMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean/Initializer/zeros:0@H
¤
;MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance:0@MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/AssignOMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Read/ReadVariableOp:0(2LMobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance/Initializer/ones:0@H*Ţ
image_feature_vectorĹ
=
images3
hub_input/images:0˙˙˙˙˙˙˙˙˙ŕŕ
MobilenetV1/Conv2d_6_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_4_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_8_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_1_depthwisea
BMobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙pp 
MobilenetV1/Conv2d_10_depthwisec
CMobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_9_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_4_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_7_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_9_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_12_pointwisec
CMobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙]
defaultR
:hub_output/feature_vector/SpatialSqueeze/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_11_pointwisec
CMobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_11_depthwisec
CMobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_13_depthwisec
CMobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_2_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙88
MobilenetV1/Conv2d_5_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_13_pointwisec
CMobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_8_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_3_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙88
MobilenetV1/Conv2d_1_pointwisea
BMobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙pp@
MobilenetV1/Conv2d_12_depthwisec
CMobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_10_pointwisec
CMobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_2_depthwisea
BMobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙88@
MobilenetV1/Conv2d_5_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙q
MobilenetV1/AvgPool_1aW
7MobilenetV1/Logits/AvgPool_1a/AvgPool/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_7_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙o
MobilenetV1/Conv2d_0W
8MobilenetV1/MobilenetV1/Conv2d_0/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙pp 
MobilenetV1/Conv2d_6_pointwiseb
BMobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙
MobilenetV1/Conv2d_3_depthwiseb
BMobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6/ReadForQuantize:0˙˙˙˙˙˙˙˙˙88*Ş
default
=
images3
hub_input/images:0˙˙˙˙˙˙˙˙˙ŕŕ]
defaultR
:hub_output/feature_vector/SpatialSqueeze/ReadForQuantize:0˙˙˙˙˙˙˙˙˙