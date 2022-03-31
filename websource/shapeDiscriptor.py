
try: 
    from utils.ZernikeMoments import ZernikeMoments
    from utils.Searcher import Searcher
    from imutils.paths import list_images
    from matplotlib import pyplot as plt
    from PIL import Image
    import pickle, imutils 
    import cv2 
    import numpy as np
except :
    print("Error occured")


def convert_bg_to_white(inp_path, out_path):
    img_path = inp_path
    for spritePath in list_images(img_path):
        #print(pokemon)
        image = cv2.imread(spritePath)
        #print(image)

        image = Image.open(spritePath).convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)

        new_image.convert("RGB").save(out_path+pokemon.split('\\')[1]+".png")

def img_to_gs_shape(img): 
        desc = ZernikeMoments(33)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # pad the image with extra white pixels to ensure the
        # edges of the pokemon are not up against the borders
        # of the image
        image = cv2.copyMakeBorder(image, 15, 15, 15, 15,
            cv2.BORDER_CONSTANT, value = 255)
        # invert the image and threshold it
        thresh = cv2.bitwise_not(image)
        thresh[thresh > 0] = 255
        #plt.imshow(thresh)
        #plt.sho
        # initialize the outline image, find the outermost
        # contours (the outline) of the pokemone, then draw
        # it
        outline = np.zeros(image.shape, dtype = "uint8")
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)
        #print(index)
        return desc.describe(outline)

def img_to_index(inp_path, out_path): 
    for spritePath in list_images(inp_path):
    # parse out the pokemon name, then load the image and
    # convert it to grayscale
        index = dict()
        pokemon = spritePath[spritePath.rfind("/") + 1:].replace(".png", "")
        image = cv2.imread(spritePath)
        index[pokemon] = img_to_gs_shape(image)
        f = open(out_path+"./index.cpickle", "wb")
        f.write(pickle.dumps(index))
        f.close()


###ANNOTATIONS -> For performance reason check if index can be init once
def find_pokemon(inp_path, idx_path):
    image = cv2.imread(inp_path)
    #cv2.imread("./greyscaled/pikachu.png")
    #cv2.imread("./pias.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = imutils.resize(image, width = 64)
    image.shape
    image = cv2.resize(image, [120,120])
    queryFeatures = img_to_gs_shape(image)
    # perform the search to identify the pokemon
    index = open(idx_path, "rb").read()
    index = pickle.loads(index)
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)
    return results


find_pokemon("./greyscaled/pikachu.png", "./index.cpickle")