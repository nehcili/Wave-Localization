�$	[rP!�@ߔ&*2��@�����@!К.�@$	_3��cB@������A@��b!3'@!Wş5��N@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�����@m��$t@A�/��e9�@Y$|�o���@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&К.�@V��;Mf�?A������@Y��l7�@*	bXɊ�ZA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�o�D�"�@!�%�oF�I@)�o�D�"�@1�%�oF�I@:Preprocessing2P
Iterator::Model::Prefetch�N?��̪@!;�bH@)�N?��̪@1;�bH@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�����'�@!��[g��I@)ۢ���@1��f0�?:Preprocessing2F
Iterator::Model5��4ͪ@!~/��jbH@)�����Q�?1�њD�i?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�J[\�3y?!�8 L�?)�J[\�3y?1�8 L�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 18.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�ikD0)d@���(vl@V��;Mf�?!m��$t@	!       "	!       *	!       2$	�+�,�Ӽ@ф2b���@�/��e9�@!������@:	!       B	!       J$	Z'.�k͚@G�%�h@$|�o���@!��l7�@R	!       Z$	Z'.�k͚@G�%�h@$|�o���@!��l7�@JCPU_ONLY2black"�
both�Your program is MODERATELY input-bound because 18.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 