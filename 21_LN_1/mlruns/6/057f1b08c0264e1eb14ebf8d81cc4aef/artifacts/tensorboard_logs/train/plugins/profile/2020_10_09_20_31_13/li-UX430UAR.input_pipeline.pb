	s/0+��@s/0+��@!s/0+��@	��q��fQ@��q��fQ@!��q��fQ@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$s/0+��@#I��du@Ake�/��o@Y}гY�f�@*	���pw%EA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorυ�^��@!�!")HI@)υ�^��@1�!")HI@:Preprocessing2P
Iterator::Model::Prefetch��T�7f�@!������H@)��T�7f�@1������H@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapd�=	L�@!�B��JI@)�C6�.�?1|=�Nb%�?:Preprocessing2F
Iterator::Model�{f�@!R���H@)�ЕT��?1���8��c?:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensorT8�T�]?!���9v?)T8�T�]?1���9v?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 69.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2B17.4 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#I��du@#I��du@!#I��du@      ��!       "      ��!       *      ��!       2	ke�/��o@ke�/��o@!ke�/��o@:      ��!       B      ��!       J	}гY�f�@}гY�f�@!}гY�f�@R      ��!       Z	}гY�f�@}гY�f�@!}гY�f�@JCPU_ONLY