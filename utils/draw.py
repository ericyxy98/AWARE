import matplotlib.pyplot as plt
from matplotlib import animation

def spectrogram(inputs, mask=None, animate=False, flipud=False):
    inputs = inputs/2+0.5
    if flipud:
        inputs = inputs.flip(dims=(3,))
    plt.rcParams["animation.html"] = "jshtml"
    d = math.isqrt(inputs.shape[0]-1)+1
    fig, axs = plt.subplots(d, d, figsize=(8,8))
    im = []
    msk = []
    if inputs.shape[0]==1:
        im += [axs.imshow(inputs[0,:,0,:,:].permute(1,2,0))]
        if mask is not None:
            msk += [axs.imshow(mask[0,:,:], cmap='jet', alpha=0.4)]
    else:
        for i in range(inputs.shape[0]):
            # im += [axs[i//d, i%d].pcolormesh(inputs[i,0,0,:,:], shading='gouraud', cmap='gray')]
            im += [axs[i//d, i%d].imshow(inputs[i,:,0,:,:].permute(1,2,0))]
            if mask is not None:
                 msk += [axs[i//d, i%d].imshow(mask[i,:,:], cmap='jet', alpha=0.4)]
    
    def update(frame):
        for i in range(inputs.shape[0]):
            im[i].set_array(inputs[i,:,frame,:,:].permute(1,2,0))
    
    if animate:
        ani = animation.FuncAnimation(fig=fig, func=update, frames=inputs.shape[2], interval=100)
        display(ani)
        ani.save('animation.gif', writer='imagemagick', fps=10)
        plt.close(fig)
    else:
        plt.show()

