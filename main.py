import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from matplotlib.widgets import Slider, Button, RadioButtons


class DrawStyle(object):
    def __init__(self, image_name):
        self.min_edge = 50
        self.max_edge = 70
        self.k_size = 3
        self.k_iter = 1

        self.image_name = image_name.split(".")[0]
        self.scale_percent = 30
        image = cv2.imread(image_name)
        self.image = image[:, :, ::-1]
        self.new_image = None
        self.edges_shown = False

    def reduce(self, img):
        """
        Reduces the size of the image for easier interactive showing
        """
        width = int(img.shape[1] * self.scale_percent / 100)
        height = int(img.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim)
        return img

    def get_edges(self):
        """
        Find the edges in the image
        """
        edges = cv2.Canny(self.image, self.min_edge, self.max_edge, L2gradient=True)
        edges = 255 - edges

        kernel = np.ones((self.k_size, self.k_size), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=self.k_iter)

        edges = edges / 255.0
        self.edges = np.expand_dims(edges, -1)
        return self.edges

    def apply_style(self):
        """
        Applies the selected style
        """
        edges = self.get_edges()
        self.new_image = self.image * edges
        random_edges = np.random.uniform(0, 255, size=np.r_[edges.shape[:2], 3])
        edges = np.bitwise_not(edges.astype(bool))
        # self.edges = random_edges * edges
        # self.new_image = self.new_image + self.edges
        self.new_image = self.new_image.astype(np.uint8)
        return self.new_image

    def main(self):
        """
        Main function.
        It makes an interactive interface to modify the image
        """
        fig, ax = plt.subplots(1, 2)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        ax[0].margins(x=0)
        ax[1].margins(x=0)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)

        ax[0].imshow(self.reduce(self.image))
        ax[0].set_title("Original")
        self.apply_style()
        ax[1].imshow(self.reduce(self.new_image))
        ax[1].set_title("Drawn")

        min_canny_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
        max_canny_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        kernel_size_ax = plt.axes([0.25, 0.20, 0.65, 0.03])
        iterations_ax = plt.axes([0.25, 0.25, 0.65, 0.03])

        self.min_canny = Slider(
            min_canny_ax,
            "Min Canny",
            valmin=0,
            valmax=255,
            valinit=self.min_edge,
            valstep=1,
        )
        self.max_canny = Slider(
            max_canny_ax,
            "Max Canny",
            valmin=0,
            valmax=255,
            valinit=self.max_edge,
            valstep=1,
        )
        self.kernel_size = Slider(
            kernel_size_ax,
            "Kernel Size",
            valmin=1,
            valmax=10,
            valinit=self.k_size,
            valstep=1,
        )
        self.iterations = Slider(
            iterations_ax,
            "Kernel Iter",
            valmin=1,
            valmax=10,
            valinit=self.k_iter,
            valstep=1,
        )

        def update(val):
            self.min_edge = int(self.min_canny.val)
            self.max_edge = int(self.max_canny.val)
            self.k_size = int(self.kernel_size.val)
            self.k_iter = int(self.iterations.val)
            self.apply_style()
            if self.edges_shown:
                ax[1].imshow(self.reduce(self.edges[:, :, 0]), cmap="gray")
            else:
                ax[1].imshow(self.reduce(self.new_image))
            fig.canvas.draw_idle()

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(resetax, "Reset", hovercolor="0.975")

        def reset(event):
            self.min_canny.reset()
            self.max_canny.reset()
            self.kernel_size.reset()
            self.iterations.reset()

        self.reset_button.on_clicked(reset)

        edgesax = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.edges_button = Button(edgesax, "Show Edges", hovercolor="0.975")

        def show_edges(event):
            if self.edges_shown:
                ax[1].imshow(self.reduce(self.new_image))
                self.edges_shown = False
            else:
                ax[1].imshow(self.reduce(self.edges[:, :, 0]), cmap="gray")
                self.edges_shown = True
            fig.canvas.draw_idle()

        self.edges_button.on_clicked(show_edges)

        saveax = plt.axes([0.4, 0.025, 0.1, 0.04])
        self.save_button = Button(saveax, "Save", hovercolor="0.975")

        def save(event):
            cv2.imwrite(
                "{}_drawn.jpg".format(self.image_name), self.new_image[:, :, ::-1]
            )
            # edges = np.repeat(self.edges, 3, axis=-1) * 255
            edges = self.edges * 255
            cv2.imwrite("{}_edges.jpg".format(self.image_name), edges.astype(np.uint8))

        self.save_button.on_clicked(save)

        self.min_canny.on_changed(update)
        self.max_canny.on_changed(update)
        self.kernel_size.on_changed(update)
        self.iterations.on_changed(update)
        plt.show()


if __name__ == "__main__":
    image_name = "ducks.jpg"
    style = DrawStyle(image_name)
    style.main()
