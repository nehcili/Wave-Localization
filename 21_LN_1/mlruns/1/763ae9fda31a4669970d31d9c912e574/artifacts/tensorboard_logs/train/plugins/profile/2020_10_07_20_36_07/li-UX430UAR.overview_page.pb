�	6[yɧY�@6[yɧY�@!6[yɧY�@	��6��X@��6��X@!��6��X@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6[yɧY�@"T����?Ad#��D@YUg��N�@*	C�l�׌EA2P
Iterator::Model::Prefetch���|1�@!��x
��X@)���|1�@1��x
��X@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�i����@!���TDn�?)�i����@1���TDn�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�#D�@!?r2���?)Q�5�U��?1�M+E^X?:Preprocessing2F
Iterator::Modelf��B�@!��洯�X@))<hv�[�?1"�,�m�S?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensord��TkaV?!��t�Z	?)d��TkaV?1��t�Z	?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 98.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	"T����?"T����?!"T����?      ��!       "      ��!       *      ��!       2	d#��D@d#��D@!d#��D@:      ��!       B      ��!       J	Ug��N�@Ug��N�@!Ug��N�@R      ��!       Z	Ug��N�@Ug��N�@!Ug��N�@JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 98.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 