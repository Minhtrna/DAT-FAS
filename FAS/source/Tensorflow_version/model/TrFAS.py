import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class L2Norm(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

class Conv_block(layers.Layer):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        # Map manual padding to standard Keras padding
        # (1, 1) usually implies 'same' for 3x3 kernel, (0, 0) is 'valid'
        pad_type = 'same' if padding[0] > 0 else 'valid'
        
        self.conv = layers.Conv2D(filters=out_c, kernel_size=kernel, strides=stride, 
                                  padding=pad_type, use_bias=False, groups=groups)
        self.bn = layers.BatchNormalization()
        # shared_axes=[1, 2] for channels_last (N, H, W, C) sharing spatial dims
        self.prelu = layers.PReLU(shared_axes=[1, 2])

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(layers.Layer):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        pad_type = 'same' if padding[0] > 0 else 'valid'
        
        self.conv = layers.Conv2D(filters=out_c, kernel_size=kernel,
                                  groups=groups, strides=stride, padding=pad_type, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(layers.Layer):
     def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

     def call(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(layers.Layer):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(modules)

    def call(self, x):
        return self.model(x)

class SEModule(layers.Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        # Global Average Pooling but keep dims (N, 1, 1, C) for multiplication
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(channels // reduction, kernel_size=1, padding='valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(channels, kernel_size=1, padding='valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x

class ResidualSE(layers.Layer):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block-1:
                modules.append(
                    Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride,
                               groups=groups, se_reduct=se_reduct))
            else:
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding,
                                          stride=stride, groups=groups))
        self.model = Sequential(modules)

    def call(self, x):
        return self.model(x)

class Depth_Wise_SE(layers.Layer):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def call(self, x):
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output

class MiniFASNet(Model):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),
                 drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]

        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]

        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]

        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]

        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]

        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = layers.Flatten()
        self.linear = layers.Dense(embedding_size, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_p)
        self.prob = layers.Dense(num_classes, use_bias=False)

    def call(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out

class MiniFASNetSE(MiniFASNet):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7),drop_p=0.75, num_classes=4, img_channel=3):
        super(MiniFASNetSE, self).__init__(keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                           drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]

        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))



keep_dict = {'1.8M': [32, 32, 103, 103, 64, 13, 13, 64, 26, 26,
                      64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
                      154, 154, 128, 52, 52, 128, 26, 26, 128, 52,
                      52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
                      128, 26, 26, 128, 26, 26, 128, 512, 512],

             '1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                       13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                       52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                       128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
             }


# (80x80) flops: 0.044, params: 0.41
def MiniFASNetV1(embedding_size=128, conv6_kernel=(7, 7),
                     drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict['1.8M'], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


# (80x80) flops: 0.044, params: 0.43
def MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7),
                     drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict['1.8M_'], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)

def MiniFASNetV1SE(embedding_size=128, conv6_kernel=(7, 7),
                   drop_p=0.75, num_classes=3, img_channel=3):
    return MiniFASNetSE(keep_dict['1.8M'], embedding_size, conv6_kernel,drop_p, num_classes, img_channel)

# (80x80) flops: 0.044, params: 0.43
def MiniFASNetV2SE(embedding_size=128, conv6_kernel=(7, 7),
                   drop_p=0.75, num_classes=4, img_channel=3):
    return MiniFASNetSE(keep_dict['1.8M_'], embedding_size, conv6_kernel,drop_p, num_classes, img_channel)


# ------------------------------------------------------------------------------
# MobileNext (Keras Implementation)
# Ported from mnext.py (PyTorch)
# ------------------------------------------------------------------------------

import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SGBlock(layers.Layer):
    """
    SandGlass Block:
    Depthwise (3x3) -> Pointwise (1x1) -> Pointwise (1x1) -> Depthwise (3x3)
    """
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio

        self.layers_list = []

        # PyTorch uses ReLU6, Keras has tf.nn.relu6. We can use Activation layers or Lambda.
        # MiniFASNet uses PReLU or ReLU, but MobileNetV2/Next generally uses ReLU6.
        # We will use ReLU6 to match MobileNext reference.
        def ReLU6():
            return layers.ReLU(max_value=6.0)

        if expand_ratio == 2:
            # dw
            self.layers_list.append(layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)) 
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())
            # pw-linear
            self.layers_list.append(layers.Conv2D(hidden_dim, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            # pw-linear
            self.layers_list.append(layers.Conv2D(oup, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())
            # dw
            # Note: stride helps here
            pad_type = 'same' if stride == 1 else 'same' # Keras 'same' padding with stride 2 handles dowsampling differently than PyTorch commonly? 
            # PyTorch padding=1 with kernel 3 is 'same'.
            self.layers_list.append(layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())

        elif inp != oup and stride == 1 and keep_3x3 == False:
            # pw-linear
            self.layers_list.append(layers.Conv2D(hidden_dim, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            # pw-linear
            self.layers_list.append(layers.Conv2D(oup, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())

        elif inp != oup and stride == 2 and keep_3x3==False:
            # pw-linear
            self.layers_list.append(layers.Conv2D(hidden_dim, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            # pw-linear
            self.layers_list.append(layers.Conv2D(oup, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())
            # dw
            self.layers_list.append(layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
        
        else:
            if keep_3x3 == False:
                self.identity = True
            
            # dw
            self.layers_list.append(layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())
            # pw
            self.layers_list.append(layers.Conv2D(hidden_dim, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            # pw
            self.layers_list.append(layers.Conv2D(oup, 1, 1, padding='valid', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(ReLU6())
            # dw
            self.layers_list.append(layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False))
            self.layers_list.append(layers.BatchNormalization())

        self.block = Sequential(self.layers_list)

    def call(self, x):
        out = self.block(x)
        if self.identity:
            # Identity connection with slicing as per PyTorch implementation
            # PyTorch: x[:,:shape[1]//self.identity_div,:,:]
            # Keras (NHWC): x[:, :, :, :shape[-1]//self.identity_div]
            shape = tf.shape(x)
            # We need static shape if possible or dynamic
            # For slicing in call, we can use slice syntax
            c = x.shape[-1]
            if c is None: # Dynamic shape
                 # Handling dynamic shape might be tricky without tf.shape, but split is usually static C
                 pass 
            
            # Based on mnext.py: out[:,:c//div] = out[...] + x[...]
            # Keras tensors are immutable. We must use simple addition if shapes match, or concat.
            # But the PyTorch code does in-place assignment to a slice: out[:, :c//div] += x[:, :c//div]
            
            div = self.identity_div
            # Since Keras is channels-last
            input_sliced = x[..., :x.shape[-1]//div]
            out_sliced = out[..., :out.shape[-1]//div]
            out_rest = out[..., out.shape[-1]//div:]
            
            # We add them
            added = out_sliced + input_sliced
            
            # Concatenate back
            out = tf.concat([added, out_rest], axis=-1)
            
        return out


class MobileNext(Model):
    def __init__(self, width_mult=1.0, **kwargs):
        super(MobileNext, self).__init__(**kwargs)
        
        # CIFAR 10 and 100 low parameter version (from mnext.py)
        # [t, c, n, s]
        self.cfgs = [
            [2,   64, 1, 1],
            [6,   96, 1, 1],
            [6,  128, 3, 2],
            [6,  192, 2, 1],
            [6,  256, 3, 2],
            [6,  384, 2, 1],
            [6,  512, 3, 2], # Changed to stride 2 to match MiniFASNet reduction
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        
        # conv_3x3_bn(3, input_channel, 1) -> but in mnext it isstride 1 for CIFAR
        self.stem = Sequential([
            layers.Conv2D(input_channel, 3, strides=2, padding='same', use_bias=False), # Changed to stride 2
            layers.BatchNormalization(),
            layers.ReLU(max_value=6.0)
        ])

        self.blocks = []
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            # Note: mnext code has specific logic for c=1280, skipping here as cfg ends at 512
            
            # First block in sequence handles stride
            self.blocks.append(block(input_channel, output_channel, s, t, keep_3x3=(n==1 and s==1)))
            input_channel = output_channel
            
            # Subsequent blocks are stride 1
            for i in range(n-1):
                self.blocks.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        
        self.features_seq = Sequential(self.blocks)
        self.out_channels = input_channel # Store final output channel count

    def call(self, x):
        x = self.stem(x)
        x = self.features_seq(x)
        return x


class MobileNextFASNet(Model):
    """
    FASNet model using MobileNext as backbone.
    Replaces MiniFASNet backbone with MobileNext feature extractor.
    """
    def __init__(self, embedding_size=128, num_classes=3, drop_p=0.2, width_mult=1.0, conv6_kernel=(5, 5)):
        super(MobileNextFASNet, self).__init__()
        self.embedding_size = embedding_size
        
        # Backbone
        self.backbone = MobileNext(width_mult=width_mult)
        backbone_out_channels = self.backbone.out_channels
        
        # FAS Head (matching MiniFASNet structure)
        # keep[46] -> keep[47] -> keep[48] ...
        # logic in MiniFASNet:
        # conv_6_sep (1x1)
        # conv_6_dw (kernel=conv6_kernel group=groups)
        # flatten -> linear -> bn -> drop -> prob
        
        # We need to define some channel sizes for the head.
        # MiniFASNet last output from conv_5 is 'keep[46]'.
        # Here we just use backbone output.
        
        # Arbitrarily choosing head dimensions similar to 1.8M config end
        # keep[47] = 128
        # keep[48] = 512
        head_dim_1 = 512
        head_dim_2 = 512
        
        self.conv_6_sep = Conv_block(backbone_out_channels, head_dim_1, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(head_dim_1, head_dim_2, groups=head_dim_2, kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = layers.Flatten()
        
        self.linear = layers.Dense(embedding_size, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_p)
        self.prob = layers.Dense(num_classes, use_bias=False)

    def call(self, x):
        # Backbone features
        x = self.backbone(x) # (N, H', W', C)
        
        # Head
        x = self.conv_6_sep(x)
        # Note: conv6_kernel is (7,7) or similar. 
        # If backbone output spatial size < 7x7, this might fail or be valid padding dependent.
        # MiniFASNet input 80x80 -> reduced to roughly 7x7 or 5x5.
        # MobileNext with stride 2 at some stages:
        # 80 -> 80 (stem stride 1) -> 80 (stage 1) -> 80 (stage 2) -> 40 (stage 3) -> 40 -> 20 (stage 4) -> 20 -> 20 (stage 6?)
        # Let's trace MobileNext strides:
        # cfgs: [s=1, s=1, s=2, s=1, s=2, s=1, s=1]
        # 80 -> 80 -> 80 -> 40 -> 40 -> 20 -> 20 -> 20
        # Final output is 20x20.
        # MiniFASNet conv6_kernel=(7,7). 
        # If we use valid padding on 20x20 with 7x7 kernel -> 14x14.
        
        x = self.conv_6_dw(x) 
        x = self.conv_6_flatten(x)
        
        if self.embedding_size != 512:
            x = self.linear(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.prob(x)
        return x


if __name__ == "__main__":
    # Test block as requested
    try:
        print("Testing MobileNextFASNet instantiation...")
        model = MobileNextFASNet(embedding_size=128, num_classes=3, width_mult=1.0)
        
        # Create a dummy input
        dummy_input = tf.random.normal((1, 80, 80, 3))
        
        # Run forward pass
        output = model(dummy_input)
        
        print("Model built successfully.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        model.summary()
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
