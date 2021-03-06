import matplotlib.pyplot as plt

def plot_acc_loss(history):
    """ Plot the loss and accuray for the training and validation"""
    # Get the values from the model history
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # plot the accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    
    # Plot the loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    

def smooth_curve(points, factor = 0.8 ):
    """Calculate the exonential moving average considering 1 lag"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+ point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points    
    

def plot_acc_loss_smooth(history):
    """ Plot the smooth loss and accuray for the training and validation.
    For smoothing the values we use the smooth_curve function"""
    # Get the values from the model history
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # plot the accuracy
    plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    
    # Plot the loss
    plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    

def deprocess_image(x):
    """ Function to normalize the values of the image"""
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x,0,1)
    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size = 150):
    """This function create the image which maximize the specify filter"""
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) 
    iterate = K.function([model.input],[loss, grads])
    input_img_data = np.random.random((1,150,150,3)) * 20 +128
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


def plot_filters (model,layer_name):
    """This function print 64 filters for the specified layer"""
    size = 150
    margin = 5
    results = np.zeros((8* size + 7 * margin, 8 * size + 7* margin,3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(model, layer_name, i + (j*8), size = size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end,:] = filter_img
    plt.figure(figsize = (20,20))
    results = results.astype(int)
    plt.imshow(results)