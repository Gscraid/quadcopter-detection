import cv2 as cv

def visual(bboxes, image, scores,labels):
	k = 0
	for i in range(scores.size):
		if(scores[i] > 0):
			x1 = int(bboxes[i][0])
			y1 = int(bboxes[i][1])
			x2 = int(bboxes[i][2])
			y2 = int(bboxes[i][3])
			if(labels[i]==1):
				im = cv.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(50, 208, 82),3)
				cv.putText(im, "Quadro: %.2f" %scores[i], (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 208, 82), 2)
			k = 1
	if(k==1):
		return im
	else:
		return image