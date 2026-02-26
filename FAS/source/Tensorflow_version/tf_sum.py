import tensorflow as tf
from tensorflow import keras
from Model.MultiFTNet import MultiFTNet
from Model.MiniFASNet import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2SE


# MXNet backbone (hiện tại trong MultiFTNet)
model_mnext = MultiFTNet(num_classes=2, img_channel=3, embedding_size=128)

# Build model với input shape
dummy_input = tf.random.normal((1, 224, 224, 3))  # Điều chỉnh kích thước nếu cần
_ = model_mnext(dummy_input, training=False)

# Tổng params (bao gồm FTGenerator)
params_mnext_total = model_mnext.count_params()

# Params của FTGenerator
params_ft = model_mnext.FTGenerator.count_params()
params_mnext_inference = params_mnext_total - params_ft

# MiniFASNet models
model_mini_v1 = MiniFASNetV1(num_classes=3, img_channel=3)
model_mini_v1.build(input_shape=(None, 224, 224, 3))
params_mini_v1 = model_mini_v1.count_params()

model_mini_v1se = MiniFASNetV1SE(num_classes=3, img_channel=3)
model_mini_v1se.build(input_shape=(None, 224, 224, 3))
params_mini_v1se = model_mini_v1se.count_params()

model_mini_v2se = MiniFASNetV2SE(num_classes=4, img_channel=3)
model_mini_v2se.build(input_shape=(None, 224, 224, 3))
params_mini_v2se = model_mini_v2se.count_params()

print("=" * 55)
print("So sánh số lượng tham số (inference only)")
print("=" * 55)
print(f"MultiFTNet (MXNet) - inference:  {params_mnext_inference:>10,} params")
print(f"MultiFTNet (MXNet) - total:      {params_mnext_total:>10,} params")
print(f"  └─ FTGenerator (train only):   {params_ft:>10,} params")
print(f"MiniFASNetV1:                    {params_mini_v1:>10,} params")
print(f"MiniFASNetV1SE:                  {params_mini_v1se:>10,} params")
print(f"MiniFASNetV2SE:                  {params_mini_v2se:>10,} params")