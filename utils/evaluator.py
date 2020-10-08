"""
Adapted from Josiah Wang
https://github.com/josiahwang/phraseloceval
"""


class Evaluator(object):
	"""
	Utility class for evaluating phrase localization
	"""

	def __init__(self):
		pass

	def compute_iou(self, predictedBoxList, gtBoxList):
		""" Computes list of areas of IoU for all given instances.
		Parameters
		----------
		predictedBoxList : list
			[[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
			List of predicted bounding box instances [x,y,w,h] for each query instance.
			x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
			w and h are the width and height of the bounding box for the query test
		gtBoxList : list
			Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
		Returns
		-------
		iouList : list(float)
			The area of IoU for each prediction in predictedBoxList
		"""

		assert len(predictedBoxList) == len(gtBoxList), \
			"The list of predicted bounding boxes ({}) should be the same size as the list of ground truth bounding boxes ({})." \
				.format(len(predictedBoxList), len(gtBoxList))

		# compute iou for each bounding box instance
		iouList = []
		for (box1, box2) in zip(gtBoxList, predictedBoxList):
			iou = self._iou(box1, box2)
			iouList.append(iou)

		return iouList

	def accuracy(self, iouList, iouThreshold = 0.5):
		""" Computes the overall accuracy from a given list of iou and an iouThreshold
		Parameters
		----------
		iouList : list(float)
			List of areas of IoU
		iouThreshold : float
			The threshold for the IoU, such that item in iouList is True if IoU >= iouThreshold.
		Returns
		-------
		accuracy : float
			Overall accuracy (or recall to be more precise).
			Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

		"""

		matches = len([1 for iou in iouList if iou >= iouThreshold]);
		accuracy = matches * 1.0 / len(iouList);
		return accuracy;

	def evaluate(self, predictedBoxList, gtBoxList, iouThreshold = 0.5):
		""" Computes the overall accuracy and list of areas of IoU for each test instance.
		Parameters
		-------
		predictedBoxList : list
			[[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
			List of predicted bounding box instances [x,y,w,h] for each query instance.
			x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
			w and h are the width and height of the bounding box for the query test
		gtBoxList : list
			Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
		iouThreshold : float
			The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
		Returns
		-------
		accuracy : float
			Overall accuracy (or recall to be more precise).
			Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

		iouList : list(float)
			The area of IoU for each prediction in predictedBoxList
		"""

		iouList = self.compute_iou(predictedBoxList, gtBoxList);
		accuracy = self.accuracy(iouList, iouThreshold);
		return (accuracy, iouList);

	def evaluate_perclass(self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold = 0.5):
		""" Computes the overall accuracy, per-category accuracies, and list of areas of IoU for each test instance.
		Parameters
		----------
		predictedBoxList : list
			[[x,y,w,h],[x,y,w,h],[x,y,w,h],...]
			List of predicted bounding box instances [x,y,w,h] for each query instance.
			x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
			w and h are the width and height of the bounding box for the query test
		gtBoxList : list
			Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
		iouThreshold : float
			The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
		boxCategoriesList : list of list
			List of categories per box instance. Each box can have more than one category. Must be the same length as gtBoxList
		Returns
		-------
		accuracy : float
			Overall accuracy (or recall to be more precise).
			Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

		perclassAccuracies : dict
			Per-class accuracy. Key: category label; Value: accuracy (float).
		iouList : list(float)
			The area of IoU for each prediction in predictedBoxList
		"""

		# get all categories
		categorySet = set();
		for categoryList in boxCategoriesList:
			categorySet.update(categoryList);

		iouList = self.compute_iou(predictedBoxList, gtBoxList);
		accuracy = self.accuracy(iouList, iouThreshold);

		# evaluate for each category
		perClassAccDict = {};
		for category in categorySet:
			# get subset of instances for this category
			subPredictedBoxList = [];
			subGtBoxList = [];
			for (pred, gt, categoryList) in zip(predictedBoxList, gtBoxList, boxCategoriesList):
				if category in categoryList:
					subPredictedBoxList.append(pred);
					subGtBoxList.append(gt);
			# print("{}: {}".format(category, len(subGtBoxList)));

			# and evaluate subset
			subIouList = self.compute_iou(subPredictedBoxList, subGtBoxList);
			perClassAccDict[category] = self.accuracy(subIouList, iouThreshold);

		return (accuracy, perClassAccDict, iouList);

	def evaluate_upperbound_perclass(self, predictedBoxList, gtBoxList, boxCategoriesList, iouThreshold = 0.5):
		""" Computes the overall accuracy, per-category accuracies, and list of areas of IoU for each test instance.
		Assumes that there are multiple candidate bounding boxes per test instance in predictedBoxList,
		and we keep the max iou across all candidates to get the best iou per test instance
		Parameters
		----------
		predictedBoxList : list of list
			[[[x,y,w,h],[x,y,w,h]],[[x,y,w,h],[x,y,w,h]],...]
			List of predicted bounding box instances [x,y,w,h] for each query instance.
			x and y are the (x,y) coordinates of the top-left of the bounding box for the query term
			w and h are the width and height of the bounding box for the query test
		gtBoxList : list
			Same as above, but for ground truth bounding boxes. Must be the same length as predictedBoxList
		iouThreshold : float
			The threshold for the IoU, such that two bounding boxes are considered overlapping when IoU >= iouThreshold.
		boxCategoriesList : list of list
			List of categories per box instance. Each box can have more than one category. Must be the same length as gtBoxList
		Returns
		-------
		accuracy : float
			Overall accuracy (or recall to be more precise).
			Proportion of predicted boxes that overlaps with the ground truth boxes by an IoU of >= iouThreshold.

		perclassAccuracies : dict
			Per-class accuracy. Key: category label; Value: accuracy (float).
		iouList : list(float)
			The area of max IoU for each prediction set in predictedBoxList
		argmaxList : list(int)
			The index of the box that maximizes the IoU for each prediction set in predictedBoxList
		"""

		# get all categories
		categorySet = set();
		for categoryList in boxCategoriesList:
			categorySet.update(categoryList);

		iouList = [];
		argmaxList = [];
		for (i, gtBox) in enumerate(gtBoxList):
			# replicate gt boxes to be the same number as number of candidates in prediction
			nCandidates = len(predictedBoxList[i]);
			replicatedGtBoxList = [];
			for j in range(nCandidates):
				replicatedGtBoxList.append(gtBox);
			instanceIouList = self.compute_iou(predictedBoxList[i], replicatedGtBoxList);
			maxIou = max(instanceIouList);
			iouList.append(maxIou);
			argmaxList.append(instanceIouList.index(maxIou));
		accuracy = self.accuracy(iouList, iouThreshold);

		# evaluate for each category
		perClassAccDict = {};
		for category in categorySet:
			# get subset of instances for this category
			subPredictedBoxList = [];
			subGtBoxList = [];
			for (pred, gt, categoryList) in zip(predictedBoxList, gtBoxList, boxCategoriesList):
				if category in categoryList:
					subPredictedBoxList.append(pred);
					subGtBoxList.append(gt);
			# print("{}: {}".format(category, len(subGtBoxList)));

			# and evaluate subset
			subIouList = [];
			for (i, subGtBox) in enumerate(subGtBoxList):
				# replicate gt boxes to be the same number as number of candidates in prediction
				nCandidates = len(subPredictedBoxList[i]);
				replicatedGtBoxList = [];
				for j in range(nCandidates):
					replicatedGtBoxList.append(subGtBox);
				instanceIouList = self.compute_iou(subPredictedBoxList[i], replicatedGtBoxList);
				maxIou = max(instanceIouList);
				subIouList.append(maxIou);

			perClassAccDict[category] = self.accuracy(subIouList, iouThreshold);

		return (accuracy, perClassAccDict, iouList, argmaxList);

	def _iou(self, box1, box2):
		"""Computes intersection over union (IoU) for two boxes.
		where each box = [x, y, w, h]
		Parameters
		----------
		box1 : list
			[x, y, w, h] of first box
		box2 : list
			[x, y, w, h] of second box
		Returns
		-------
		float
			intersection over union for box1 and box2

		"""

		(box1_left_x, box1_top_y, box1_right_x, box1_bottom_y) = box1
		box1_w = box1_right_x - box1_left_x + 1
		box1_h = box1_bottom_y - box1_top_y + 1

		(box2_left_x, box2_top_y, box2_right_x, box2_bottom_y) = box2
		box2_w = box2_right_x - box2_left_x + 1
		box2_h = box2_bottom_y - box2_top_y + 1

		# get intersecting boxes
		intersect_left_x = max(box1_left_x, box2_left_x);
		intersect_top_y = max(box1_top_y, box2_top_y);
		intersect_right_x = min(box1_right_x, box2_right_x);
		intersect_bottom_y = min(box1_bottom_y, box2_bottom_y);

		# compute area of intersection
		# the "0" lower bound is to handle cases where box1 and box2 don't overlap
		overlap_x = max(0, intersect_right_x - intersect_left_x + 1);
		overlap_y = max(0, intersect_bottom_y - intersect_top_y + 1);
		intersect = overlap_x * overlap_y;

		# get area of union
		union = (box1_w * box1_h) + (box2_w * box2_h) - intersect;

		# return iou
		return intersect * 1.0 / union;

