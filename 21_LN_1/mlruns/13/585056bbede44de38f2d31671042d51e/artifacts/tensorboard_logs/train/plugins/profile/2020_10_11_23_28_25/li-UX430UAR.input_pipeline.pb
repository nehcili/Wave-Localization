	I*Ś��@I*Ś��@!I*Ś��@	m���EDM@m���EDM@!m���EDM@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$I*Ś��@���+yw@A�n���E@YJ�i;��@*	����O2A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�x軻��@!)�Ҋ�NI@)�x軻��@1)�Ҋ�NI@:Preprocessing2P
Iterator::Model::Prefetch�N$�*�@!1z/�2�H@)�N$�*�@11z/�2�H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapǡ~v��@!-wX�FUI@)F]k�S��?1f����?:Preprocessing2F
Iterator::Model������@!ӈ�{��H@)8���C�?1�8�/�p?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensore��7id?!�f���*?)e��7id?1�f���*?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 58.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2B37.1 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���+yw@���+yw@!���+yw@      ��!       "      ��!       *      ��!       2	�n���E@�n���E@!�n���E@:      ��!       B      ��!       J	J�i;��@J�i;��@!J�i;��@R      ��!       Z	J�i;��@J�i;��@!J�i;��@JCPU_ONLY