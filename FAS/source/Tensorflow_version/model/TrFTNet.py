import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from TrFAS import MiniFASNetV2SE


class FTGenerator(layers.Layer):
    def __init__(self, in_channels=48, out_channels=1, **kwargs):
        super(FTGenerator, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ft = Sequential([
            layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                         kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                         kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same',
                         kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=None):
        return self.ft(x, training=training)


class MultiFTNet(Model):
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5), **kwargs):
        super(MultiFTNet, self).__init__(**kwargs)
        self.img_channel = img_channel
        self.num_classes = num_classes
        self._embedding_size = embedding_size
        
        # Đổi tên thành backbone để tránh conflict với Keras reserved attributes
        self.backbone = MiniFASNetV2SE(
            embedding_size=embedding_size, 
            conv6_kernel=conv6_kernel,
            num_classes=num_classes, 
            img_channel=img_channel
        )
        
        self.ft_generator = FTGenerator(in_channels=128)

    def call(self, x, training=None):
        # Forward pass qua backbone layers - GIỐNG HỆT PyTorch
        x = self.backbone.conv1(x)
        x = self.backbone.conv2_dw(x)
        x = self.backbone.conv_23(x)
        x = self.backbone.conv_3(x)
        x = self.backbone.conv_34(x)
        x = self.backbone.conv_4(x)
        
        # Tap point cho FTGenerator (sau conv_4)
        ft_input = x
        
        x1 = self.backbone.conv_45(x)
        x1 = self.backbone.conv_5(x1)
        x1 = self.backbone.conv_6_sep(x1)
        x1 = self.backbone.conv_6_dw(x1)
        x1 = self.backbone.conv_6_flatten(x1)
        
        # PyTorch LUÔN gọi linear, không có điều kiện
        x1 = self.backbone.linear(x1)
        
        x1 = self.backbone.bn(x1, training=training)
        x1 = self.backbone.drop(x1, training=training)  # Cần training flag
        cls = self.backbone.prob(x1)

        if training:
            ft = self.ft_generator(ft_input, training=training)
            return cls, ft
        else:
            return cls
    
    def get_config(self):
        config = super(MultiFTNet, self).get_config()
        config.update({
            'img_channel': self.img_channel,
            'num_classes': self.num_classes,
            'embedding_size': self._embedding_size,
        })
        return config


if __name__ == "__main__":
    try:
        print("Testing MultiFTNet instantiation...")
        model = MultiFTNet(img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5))
        
        # Create dummy input (80x80 là size chuẩn cho MiniFASNet)
        dummy_input = tf.random.normal((1, 80, 80, 3))
        
        # Build model bằng forward pass
        print("Building model...")
        cls_out, ft_out = model(dummy_input, training=True)
        
        # Summary với nested layers
        print("\n" + "="*60)
        model.summary(expand_nested=True)
        print("="*60)
        
        # Test inference
        print("\nRunning inference...")
        cls_out = model(dummy_input, training=False)
        print(f"Inference Output shape: {cls_out.shape}")
        
        # Test training
        print("Running training forward pass...")
        cls_out, ft_out = model(dummy_input, training=True)
        print(f"Training Output shapes - CLS: {cls_out.shape}, FT: {ft_out.shape}")
        
        print(f"\nTotal parameters: {model.count_params():,}")
        print("Model built successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
