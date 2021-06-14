from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dataset_extractor import load_dataset

if __name__ == "__main__":
    batch_size = 32

    # load dataset
    train_generator, test_generator = load_dataset(batch_size)

    t1, l1 = next(train_generator)
    print(t1.shape)

    # load MobileNetV2 model
    # option 1 - Random weights, train all layers
    base_model = MobileNetV2(include_top=False, input_shape=(32, 32, 3), weights=None)

    for layer in base_model.layers:
        layer.trainable = True

    # option 2 - imagenet weights, train last layer
    # base_model = MobileNetV2(include_top=False, input_shape=(32, 32, 3), weights='imagenet')
    #
    # for layer in base_model.layers:
    #     layer.trainable = False

    # add a 10 neuron dense layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(10, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # training procdure
    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

    model.fit_generator(train_generator, validation_data=test_generator, steps_per_epoch=len(train_generator)//batch_size, epochs=20)