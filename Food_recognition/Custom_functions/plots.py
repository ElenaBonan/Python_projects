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