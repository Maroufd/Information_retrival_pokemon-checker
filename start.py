from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
	# return the chi-squared distance
	return d


def shape_descriptor():
	image = cv2.imread(args["query"])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width = 64)
	# threshold the image
	thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
		cv2.THRESH_BINARY_INV, 11, 7)
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

def zernike_moment():
	desc = ZernikeMoments(21)
	queryFeatures = desc.describe(outline)
	# perform the search to identify the pokemon
	searcher = Searcher(index)
	results = searcher.search(queryFeatures)
	print "That pokemon is: %s" % results[0][1].upper()
	# show our images
	cv2.imshow("image", image)
	cv2.imshow("outline", outline)
	cv2.waitKey(0)
	return 0
def compute_distance():
	# METHOD #2: UTILIZING SCIPY
	# initialize the scipy methods to compaute distances
	SCIPY_METHODS = (
		("Euclidean", dist.euclidean),
		("Manhattan", dist.cityblock),
		("Chebysev", dist.chebyshev))
	# loop over the comparison methods
	for (methodName, method) in SCIPY_METHODS:
		# initialize the dictionary dictionary
		results = {}
		# loop over the index
		for (k, hist) in index.items():
			# compute the distance between the two histograms
			# using the method and update the results dictionary
			d = method(index[selected], hist)
			results[k] = d
		# sort the results
		results = sorted([(v, k) for (k, v) in results.items()])[:4]
		# show the query image
		fig = plt.figure("Query")
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(images[selected])
		plt.axis("off")
		# initialize the results figure
		fig = plt.figure("Results: %s" % (methodName))
		fig.suptitle(methodName, fontsize = 20)
		# loop over the results
		for (i, (v, k)) in enumerate(results):
			# show the result
			ax = fig.add_subplot(1, len(results), i + 1)
			ax.set_title("%s: %.2f" % (k, v))
			plt.imshow(images[k])
			plt.axis("off")
	# show the SciPy methods
	plt.savefig('foo3.png')
	return 0

def histogram_comp():
	OPENCV_METHODS = (
		("Correlation", cv2.HISTCMP_CORREL),
		("Chi-Squared", cv2.HISTCMP_CHISQR),
		("Intersection", cv2.HISTCMP_INTERSECT),
		("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
	# loop over the comparison methods
	for (methodName, method) in OPENCV_METHODS:
		# initialize the results dictionary and the sort
		# direction
		results = {}
		reverse = False
		# if we are using the correlation or intersection
		# method, then sort the results in reverse order
		if methodName in ("Correlation", "Intersection"):
			reverse = True
	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = cv2.compareHist(index[selected], hist, method)
		results[k] = d
	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)[:4]


	# show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[selected])
	plt.axis("off")
	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)
	# loop over the results
	for (i, (v, k)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(1, len(results), i + 1)
		ax.set_title("%s: %.2f" % (k, v), fontsize = 8)
		plt.imshow(images[k])
		plt.axis("off")
	# show the OpenCV methods
	plt.savefig('foo2.png')
	return 0



def plot_histogram(image, title, mask=None):
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])
		plt.savefig('color.png')
	return 0

def plot_histogramgrey(image, title, mask=None):
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
	bgr = image[:,:,:3] # Channels 0..2
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	alpha = image[:,:,3] # Channel 3
	result = np.dstack([gray, alpha]) # Add the alpha channel
	hist = cv2.calcHist([result], [0], None, [256], [0, 256])
	plt.figure()
	plt.title("Grayscale Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	plt.plot(hist)
	plt.xlim([0, 256])
	plt.savefig('grey.png')
	return 0
# construct the argument parser and parse the arguments
def all(selected,distance_method):
	plot=cv2.imread("/home/pokemonchecker/api/images/"+selected,cv2.IMREAD_UNCHANGED)
	plot_histogram(plot,"Color Histogram")
	plot_histogramgrey(plot,"Grey Histogram")
	# initialize the index dictionary to store the image name
	# and corresponding histograms and the images dictionary
	# to store the images themselves
	index = {}
	images = {}
	# loop over the image paths
	for imagePath in glob.glob("/home/pokemonchecker/api/images/*.png"):
		# extract the image filename (assumed to be unique) and
		# load the image, updating the images dictionary
		filename = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)
		images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
			[0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		index[filename] = hist


	# initialize the results dictionary
	results = {}
	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the custom chi-squared method, then update
		# the results dictionary
		d = chi2_distance(index[selected], hist)
		results[k] = d
	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()])[:4]
	# show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images[selected])
	plt.axis("off")
	# initialize the results figure
	fig = plt.figure("Results: Custom Chi-Squared")
	fig.suptitle("Custom Chi-Squared", fontsize = 20)
	# loop over the results
	for (i, (v, k)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(1, len(results), i + 1)
		ax.set_title("%s: %.2f" % (k, v))
		plt.imshow(images[k])
		plt.axis("off")
	# show the custom method
	plt.savefig('foo4.png')
	return "ok"
