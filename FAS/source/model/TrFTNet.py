import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from TrFAS import MiniFASNetV2SE

class FTGenerator(layers.Layer):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = Sequential([
            layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(64, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        return self.ft(x)


class MultiFTNet(Model):
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5)):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        
        # Initialize the backbone model
        # We need the SE version as per PyTorch source: MiniFASNetV2SE
        self.model = MiniFASNetV2SE(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
        
        self.FTGenerator = FTGenerator(in_channels=128)

    def call(self, x, training=None):
        # We need to access internal layers of self.model.
        # Ideally, we should refactor MiniFASNet to return intermediate outputs or expose a method.
        # However, since we are mimicking the PyTorch 'forward' method which calls layers manually:
        
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        
        # This is where we tap out for FTGenerator
        ft_input = x
        
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        
        if self.model.embedding_size != 512:
             x1 = self.model.linear(x1)
             
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if training:
            ft = self.FTGenerator(ft_input)
            return cls, ft
        else:
            return cls

if __name__ == "__main__":
    # Simple Verification
    try:
        print("Testing MultiFTNet instantiation...")
        model = MultiFTNet(img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5))
        
        # Create a dummy input
        # Standard input size for MiniFASNet is often 80x80
        dummy_input = tf.random.normal((1, 80, 80, 3))
        
        # Run forward pass (Inference)
        print("Running inference...")
        cls_out = model(dummy_input, training=False)
        print(f"Inference Output shape: {cls_out.shape}")
        
        # Run forward pass (Training)
        print("Running training forward pass...")
        cls_out, ft_out = model(dummy_input, training=True)
        print(f"Training Output shapes - CLS: {cls_out.shape}, FT: {ft_out.shape}")
        
        print("Model built successfully.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
