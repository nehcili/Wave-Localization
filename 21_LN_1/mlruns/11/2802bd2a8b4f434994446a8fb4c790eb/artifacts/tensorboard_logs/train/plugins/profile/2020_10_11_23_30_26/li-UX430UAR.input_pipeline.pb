	/ܹ0�g�@/ܹ0�g�@!/ܹ0�g�@	�]'���T?�]'���T?!�]'���T?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/ܹ0�g�@���w��?A�w.0g�@Y��_cD�?*	�S�[A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�\m�2��@!,8����X@)�\m�2��@1,8����X@:Preprocessing2F
Iterator::Model���.��?!��̼OW?)�9� U�?1����YR?:Preprocessing2P
Iterator::Model::Prefetch�;���?!`6A��2?)�;���?1`6A��2?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap;R}�3��@!3C����X@)ѕT� r?1�t��\?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���w��?���w��?!���w��?      ��!       "      ��!       *      ��!       2	�w.0g�@�w.0g�@!�w.0g�@:      ��!       B      ��!       J	��_cD�?��_cD�?!��_cD�?R      ��!       Z	��_cD�?��_cD�?!��_cD�?JCPU_ONLY