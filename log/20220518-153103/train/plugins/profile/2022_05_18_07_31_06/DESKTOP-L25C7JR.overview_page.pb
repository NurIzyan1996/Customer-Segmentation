?	u??@u??@!u??@	?'溵t???'溵t??!?'溵t??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$u??@ڬ?\m???Axz?,C?@Y?1??%???*	?????)~@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?|?5^???!c??T@)?K7?A`??1?)Q/?T@:Preprocessing2F
Iterator::Model_?Qګ?!????g?&@)?z6?>??1V????"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W[?????!?h??	@)A??ǘ???1d$? |f@:Preprocessing2U
Iterator::Model::ParallelMapV2;?O??n??!4IƟ???);?O??n??14IƟ???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?k	??g??!??#?.V@)vq?-??1????1??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?I+?v?!??N<??)?I+?v?1??N<??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!???Z????)????Mbp?1???Z????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<???!ml?<?U@)-C??6j?1?c??7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?'溵t??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ڬ?\m???ڬ?\m???!ڬ?\m???      ??!       "      ??!       *      ??!       2	xz?,C?@xz?,C?@!xz?,C?@:      ??!       B      ??!       J	?1??%????1??%???!?1??%???R      ??!       Z	?1??%????1??%???!?1??%???JCPU_ONLYY?'溵t??b Y      Y@q\?YB?E@"?
both?Your program is POTENTIALLY input-bound because 29.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?43.3458% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 