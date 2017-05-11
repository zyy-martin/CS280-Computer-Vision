from resnet_v1 import *

##model
# def shallow_resnet(inputs, 
#                  num_classes=None,
#                  is_training=True,
#                  global_pool=True,
#                  output_stride=None,
#                  reuse=None,
#                  scope='resnet_v1_50'):  
# 	blocks = [
# 	  resnet_utils.Block(
# 		  'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
# 	  resnet_utils.Block(
# 		  'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
# 	  resnet_utils.Block(
# 		  'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
# 	  resnet_utils.Block(
# 		  'block4', bottleneck, [(2048, 512, 1)] * 3)
# 	]
# 	return resnet_v1(inputs, blocks, num_classes, is_training,
# 				   global_pool=global_pool, output_stride=output_stride,
# 				   include_root_block=True, reuse=reuse, scope=scope)


##model2
# def shallow_resnet(inputs, 
#                  num_classes=None,
#                  is_training=True,
#                  global_pool=True,
#                  output_stride=None,
#                  reuse=None,
#                  scope='resnet_small'):  
# 	blocks = [
# 	  resnet_utils.Block(
# 		  'block1', bottleneck, [(128, 32, 1)] * 2 + [(128, 32, 2)]), 
# 	  resnet_utils.Block(
# 		  'block2', bottleneck, [(512, 128, 1)] * 3)
# 	]
# 	return resnet_v1(inputs, blocks, num_classes, is_training,
# 				   global_pool=global_pool, output_stride=output_stride,
# 				   include_root_block=True, reuse=reuse, scope=scope)

## model3 too simple.
# def shallow_resnet(inputs, 
#                  num_classes=None,
#                  is_training=True,
#                  global_pool=True,
#                  output_stride=None,
#                  reuse=None,
#                  scope='resnet_v1_deep'):  
# 	blocks = [
# 	  resnet_utils.Block(
# 		  'block1', bottleneck, [(32, 8, 1)] * 1 + [(32, 8, 2)]),
# 	  resnet_utils.Block(
# 		  'block2', bottleneck, [(64, 16, 1)] * 1 + [(64, 16, 2)]), 
# 	  resnet_utils.Block(
# 		  'block4', bottleneck, [(256, 64, 1)] * 2)
# 	]
# 	return resnet_v1(inputs, blocks, num_classes, is_training,
# 				   global_pool=global_pool, output_stride=output_stride,
# 				   include_root_block=True, reuse=reuse, scope=scope)


## model4 
# def shallow_resnet(inputs, 
#                  num_classes=None,
#                  is_training=True,
#                  global_pool=True,
#                  output_stride=None,
#                  reuse=None,
#                  scope='resnet_v1_50'):  
# 	blocks = [
# 	  resnet_utils.Block(
# 		  'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
# 	  resnet_utils.Block(
# 		  'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]), 
# 	  resnet_utils.Block(
# 		  'block4', bottleneck, [(512, 128, 1)] * 3)
# 	]
# 	return resnet_v1(inputs, blocks, num_classes, is_training,
# 				   global_pool=global_pool, output_stride=output_stride,
# 				   include_root_block=True, reuse=reuse, scope=scope)

##model 5 
def shallow_resnet(inputs, 
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):  
	blocks = [
	  resnet_utils.Block(
		  'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
	  resnet_utils.Block(
		  'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]), 
	  resnet_utils.Block(
		  'block4', bottleneck, [(256, 64, 1)] * 2)
	]
	return resnet_v1(inputs, blocks, num_classes, is_training,
				   global_pool=global_pool, output_stride=output_stride,
				   include_root_block=True, reuse=reuse, scope=scope)