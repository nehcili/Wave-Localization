$	
F�8��@�Nv�A�@�=yt�@!T9�)���@$	��
Q�7@��qf�)@��IG��-@!;�8(�@@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�=yt�@p��s��@Az�}�n��@Y��bV�@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T9�)���@y �H��?A�����@Y3�&���@*	Vn�t^A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator���D�q�@!)����4I@)���D�q�@1)����4I@:Preprocessing2P
Iterator::Model::Prefetch
3F�@!ԡ��H@)
3F�@1ԡ��H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapv�ݑaw�@!�g�??9I@)� O!�@1m:�ち?:Preprocessing2F
Iterator::Modelwf���@!h�K���H@)=�U���?1�?&�TT\?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor�PS�'|?!V���?)�PS�'|?1V���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 19.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�CVws@�ܴi��z@y �H��?!p��s��@	!       "	!       *	!       2$	�u?��@W���
S�@z�}�n��@!�����@:	!       B	!       J$	�.Ⱦ�@8G��7�y@��bV�@!3�&���@R	!       Z$	�.Ⱦ�@8G��7�y@��bV�@!3�&���@JCPU_ONLY