# Neural Style Transfer

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Common uses for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs. Several notable mobile apps use NST techniques for this purpose, including DeepArt and Prisma. This method has been used by artists and designers around the globe to develop new artwork based on existent style(s). [Wikipedia Article About Neural Style Transfer (NST)](https://en.wikipedia.org/wiki/Neural_Style_Transfer) 

### Sample output from project

![Sample Neural Style Transfer Output](https://github.com/Egesabanci/neural-style-transfer/blob/master/neural-style-transfer-process.png)

### How to try / use this repository?

If you want to try the actual code in one shot, you can directly go to the `notebook/neural-style-transfer.ipynb` notebook. After you reach there, if you want to try the models with your own images, just drag-and-drop your images to the `images` folder and replace your image path's in the following section (beginning of the notebook - 1st cell):

```python
# rewrite the paths for your own images and base folder
BASE = "./drive/MyDrive/neural-style-transfer"
CONTENT_IMAGE_PATH = os.path.join(BASE, "images", "content.jpeg")
STYLE_IMAGE_PATH = os.path.join(BASE, "images", "style.png")
```

### How to use these codes in my own case?

There is a folder which is called `src` in this repository. Once you get there, you will encounter a well-separated and modular folder structure. Scripts' file extension is `.py`, which means you can import them directly to your own case.