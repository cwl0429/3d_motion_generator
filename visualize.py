import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from processing import joint, jointChain

class AnimePlot():
    def __init__(self, fps):
        self.fig = plt.figure()
        self.ax = []
        self.fps = fps
    
    def set_fig(self, labels, save_path, scale = 2.5):
        self.scale = scale
        self.save_path = save_path
        for i in range(len(labels)):
            self.ax.append(self.fig.add_subplot(1, len(labels), i+1, projection="3d"))
            self.ax[i].set_title(labels[i])
        self.time_text = self.fig.text(.5, .3, "0", ha="center")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

    def set_data(self, data, frame_num=300):
        self.frame_num = frame_num
        if len(data[0]) < self.frame_num:
            self.frame_num = len(data[0])
        for i, _ in enumerate(data):
            data[i] = data[i].reshape(data[i].shape[0], int(data[i].shape[1]/3), 3)
            data[i] = data[i]*self.scale
        self.data = data


    def ani_init(self):
        for figure in self.ax:
            figure.set_xlabel('x')
            figure.set_ylabel('y')
            figure.set_zlabel('z')
            figure.set_xlim(-.8*self.scale, .8*self.scale)
            figure.set_ylim(-.8*self.scale, .8*self.scale)
            figure.set_zlim(-.8*self.scale, .8*self.scale)
            figure.axis('off') #hide axes
            figure.view_init(elev=300,azim=-90)
    
    def ani_update(self, i):
        for figure in self.ax:
            figure.lines.clear()
            figure.collections.clear()
        for f, motion in enumerate(self.data):
            for chain in jointChain:
                pre_node = joint[chain[0]]
                next_node = joint[chain[1]]
                x = np.array([motion[i, pre_node, 0], motion[i, next_node, 0]])
                y = np.array([motion[i, pre_node, 1], motion[i, next_node, 1]])
                z = np.array([motion[i, pre_node, 2], motion[i, next_node, 2]])
                if chain in jointChain[-6:]:
                    # right
                    self.ax[f].plot(x, y, z, color="#3498db")
                else:
                    #left
                    self.ax[f].plot(x, y, z, color="#e74c3c")
        self.time_text.set_text(str(i))
    
    def animate(self):
        self.anime = animation.FuncAnimation(self.fig, self.ani_update, self.frame_num, interval=1,init_func=self.ani_init)
        f = f"{self.save_path}.gif"
        writergif = animation.PillowWriter(fps = self.fps)
        self.anime.save(f, writer=writergif)
        # writervideo = animation.FFMpegWriter(fps = 10)
        # f = f"{self.save_path}.mp4"
        # self.anime.save(f, writer=writervideo)