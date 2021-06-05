def obtain_weights_vector(model):
    weights_vector = []
    weights = model.parameters()
    for weight in weights:
        print(weight.data)
        weights_vector.append(weight)
    # print(weights_vector)