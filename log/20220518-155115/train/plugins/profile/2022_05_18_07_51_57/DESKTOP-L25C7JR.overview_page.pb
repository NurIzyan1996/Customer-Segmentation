?	??e???@??e???@!??e???@	ե?	0(??ե?	0(??!ե?	0(??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??e???@3ı.n???A&S?@Y??ݓ????*	?????T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ZӼ???!N+??d?A@)?~j?t???1????=@:Preprocessing2U
Iterator::Model::ParallelMapV2??Pk?w??!۴??I1@)??Pk?w??1۴??I1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?&S???!}??^?6@)???S㥋?1????0@:Preprocessing2F
Iterator::Model??@??ǘ?!D???x>@)?0?*??1f0
84?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Ƭ?!/?Q??yQ@)?HP?x?1lӊ{'Y@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU???N@s?!?N??b@)U???N@s?1?N??b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??r?!????@)/n??r?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??JY?8??!G??s?:@)y?&1?l?1%K!?i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ե?	0(??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3ı.n???3ı.n???!3ı.n???      ??!       "      ??!       *      ??!       2	&S?@&S?@!&S?@:      ??!       B      ??!       J	??ݓ??????ݓ????!??ݓ????R      ??!       Z	??ݓ??????ݓ????!??ݓ????JCPU_ONLYYե?	0(??b Y      Y@q?~Ȑ,@"?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?14.0488% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 