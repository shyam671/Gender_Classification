from matplotlib import pyplot as plt 
#import cv2
def plot(images, **kwargs):

	#images = [cv2.resize(images[i], (100, 100), interpolation=cv2.INTER_CUBIC) for i in range(len(images))]
	outfile = kwargs["output"]

	if "predictions" in kwargs:
		# print(predict)
		predict = kwargs["predictions"]
		predict = ['female' if predict[i] < 0 else 'male' for i in range(len(predict))]
	for i,eig in enumerate(images[:6]):
		if i<3:
			plt.subplot(1,3,i+1), plt.imshow(eig, cmap=plt.cm.gray), plt.xticks([]), plt.yticks([])
			if "predictions" in kwargs:
				plt.title(predict[i])
		if i<6:
			plt.subplot(2,3,i+1), plt.imshow(eig, cmap=plt.cm.gray), plt.xticks([]), plt.yticks([])
			if "predictions" in kwargs:
				plt.title(predict[i])
		plt.savefig(outfile)

