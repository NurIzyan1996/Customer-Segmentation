	?]K??@?]K??@!?]K??@	5?X?E???5?X?E???!5?X?E???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?]K??@z6?>W[??AQ?|a2@Y??????*	gffff?_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatu????!َ2??nG@)K?46??1]t?E?E@:Preprocessing2F
Iterator::Model?ܵ?|У?!? iCT>@)B>?٬???1?g?1?-6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ?????!B҆?R'-@)-C??6??1*u?$@:Preprocessing2U
Iterator::Model::ParallelMapV2??_?L??!2??n
M @)??_?L??12??n
M @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipRI??&¶?!9Ʒ%?jQ@)ŏ1w-!?1"?e??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????w?!/?袋.@)?????w?1/?袋.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!?ےw@)U???N@s?1?ےw@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?t???!f???z?0@)?????g?1/?袋.@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95?X?E???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z6?>W[??z6?>W[??!z6?>W[??      ??!       "      ??!       *      ??!       2	Q?|a2@Q?|a2@!Q?|a2@:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????JCPU_ONLYY5?X?E???b 