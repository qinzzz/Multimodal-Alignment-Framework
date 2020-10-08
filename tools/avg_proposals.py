import json

if __name__ == "__main__":
	f = open("dataset/train_dataset.json", "r")
	det = json.load(f)

	proposals = 0.
	count = 0.
	for key, dic in det.items():
		count += 1
		proposals += len(det[key]["classes"])
		if count > 1000:
			break

	print("average proposals: ", proposals / count)
