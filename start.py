from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

fileslocations="/home/pokemonchecker/public_html/pokemonapi/"
# Ref https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
#Method 2
#Computing distance with two built in distance metric
def compute_distance(histogram_method,methodName,index,selected,images):
	if methodName=="Euclidean":
		method=dist.euclidean
	elif methodName=="Manhattan":
		method=dist.cityblock
	results = {}
	# go through of the indexes and use the given method
	for (k, hist) in index.items():
		d = method(index[selected], hist)
		results[k] = d
	results = sorted([(v, k) for (k, v) in results.items()])[:4]
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[selected])
	plt.axis("off")
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)
	for (i, (v, k)) in enumerate(results):
		ax = fig.add_subplot(1, len(results), i + 1)
		ax.set_title("%s: %.2f" % (k, v), fontsize = 8)
		plt.imshow(images[k])
		plt.axis("off")
	plt.savefig(fileslocations+histogram_method+'_'+methodName+'_'+selected)
	return 0
# Ref https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
#Method 1
#Comperation of hsitogram
def histogram_comp(histogram_method,methodName,index,selected,images):
	results = {}
	if methodName=="Chi-Squared":
		method=cv2.HISTCMP_CHISQR
	elif methodName=="Hellinger":
		method=cv2.HISTCMP_BHATTACHARYYA
	for (k, hist) in index.items():
		d = cv2.compareHist(index[selected], hist, method)
		results[k] = d
	results = sorted([(v, k) for (k, v) in results.items()], reverse = False)[:4]
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[selected])
	plt.axis("off")
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)
	for (i, (v, k)) in enumerate(results):
		ax = fig.add_subplot(1, len(results), i + 1)
		ax.set_title("%s: %.2f" % (k, v), fontsize = 8)
		plt.imshow(images[k])
		plt.axis("off")
	plt.savefig(fileslocations+histogram_method+'_'+methodName+'_'+selected)
	return 0
# distance for shape descriptor https://pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/ still TODO:
# and https://pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/
def zernike_moment():
	desc = ZernikeMoments(21)
	queryFeatures = desc.describe(outline)
	# perform the search to identify the pokemon
	searcher = Searcher(index)
	results = searcher.search(queryFeatures)
	print("That pokemon is: %s" % results[0][1].upper())
	# show our images
	cv2.imshow("image", image)
	cv2.imshow("outline", outline)
	cv2.waitKey(0)
	return 0
#plot the color histograms source: https://pyimagesearch.com/2014/05/19/building-pokedex-python-comparing-shape-descriptors-opencv/
def plot_histogram(image, title, mask=None):
	tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
	b, g, r,a = cv2.split(image)
	rgba = [b,g,r] # removing background by method 1 https://stackoverflow.com/questions/40527769/removing-black-background-and-make-transparent-from-grabcut-output-in-python-ope
	dst = cv2.merge(rgba,3)
	chans = cv2.split(dst)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])
		plt.savefig(fileslocations+'color.png')
	return 0
#plot the grey histograms based on color source: https://pyimagesearch.com/2014/05/19/building-pokedex-python-comparing-shape-descriptors-opencv/
def plot_histogramgrey(image, title, mask=None):
	bgr = image[:,:,:3]
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	alpha = image[:,:,3] #
	result = np.dstack([gray, alpha]) # removing background by method 2 https://stackoverflow.com/questions/53380318/problem-about-background-transparent-png-format-opencv-with-python
	hist = cv2.calcHist([result], [0], None, [256], [0, 256])
	plt.figure()
	plt.title("Grayscale Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	plt.plot(hist)
	plt.xlim([0, 256])
	plt.savefig(fileslocations+'grey.png')
	return 0

def shape_descriptor(images,selected):
	image = cv2.imread(images[selected])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width = 64)
	# threshold the image
	thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
		cv2.THRESH_BINARY_INV, 11, 7)
	print(tresh)
	# initialize the outline image, find the outermost
	# contours (the outline) of the pokemon, then draw
	# it
	outline = np.zeros(image.shape, dtype = "uint8")
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	cv2.drawContours(outline, [cnts], -1, 255, -1)
	return 0
# construct the argument parser and parse the arguments
def all(selected,histogram_method,compare_method,distance_method):
	plot=cv2.imread("/home/pokemonchecker/api/images/"+selected,cv2.IMREAD_UNCHANGED)
	plot_histogram(plot,"Color Histogram")
	plot_histogramgrey(plot,"Grey Histogram")
	# initialize the index dictionary to store the image name
	# and corresponding histograms and the images dictionary
	# to store the images themselves
	index = {}
	images = {}
	# loop over the image paths
	if histogram_method=='color':
		for imagePath in glob.glob("/home/pokemonchecker/api/images/*.png"):
			filename = imagePath[imagePath.rfind("/") + 1:]
			image = cv2.imread(imagePath)
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
				[0, 256, 0, 256, 0, 256])
			hist = cv2.normalize(hist, hist).flatten()
			index[filename] = hist
	elif histogram_method=='grey':
		for imagePath in glob.glob("/home/pokemonchecker/api/images/*.png"):
			filename = imagePath[imagePath.rfind("/") + 1:]
			image = cv2.imread(imagePath)
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			alpha = image[:,:,3] # Channel 3
			result = np.dstack([gray, alpha]) # Add the alpha channel
			hist = cv2.calcHist([result], [0], None, [256], [0, 256])
			hist = cv2.normalize(hist, hist).flatten()
			index[filename] = hist
	elif histogram_method=='shape':
		for imagePath in glob.glob("/home/pokemonchecker/api/images/*.png"):
			filename = imagePath[imagePath.rfind("/") + 1:]
			image = cv2.imread(imagePath)
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			alpha = image[:,:,3] # Channel 3
			result = np.dstack([gray, alpha]) # Add the alpha channel
			hist = cv2.calcHist([result], [0], None, [256], [0, 256])
			hist = cv2.normalize(hist, hist).flatten()
			index[filename] = hist

	histogram_comp(histogram_method,compare_method,index,selected,images)
	compute_distance(histogram_method,distance_method,index,selected,images)
	#shape_descriptor(images,selected)
	# initialize the results dictionary

	return "ok"
