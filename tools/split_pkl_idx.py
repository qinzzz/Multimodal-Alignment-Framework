import pickle


def split():
	vgg_imgid2idx = pickle.load(open("data/flickr30k/maf_imgid2idx.pkl", "rb"))
	vgg_train_imgid2idx = {}
	vgg_test_imgid2idx = {}
	vgg_val_imgid2idx = {}

	train_imgid2idx = pickle.load(open("data/flickr30k/train_imgid2idx.pkl", "rb"))
	test_imgid2idx = pickle.load(open("data/flickr30k/test_imgid2idx.pkl", "rb"))
	val_imgid2idx = pickle.load(open("data/flickr30k/val_imgid2idx.pkl", "rb"))

	for imgid, idx in train_imgid2idx.items():
		imgid_jpg = imgid
		vgg_train_imgid2idx[imgid] = vgg_imgid2idx[imgid_jpg]

	for imgid, idx in test_imgid2idx.items():
		imgid_jpg = imgid
		vgg_test_imgid2idx[imgid] = vgg_imgid2idx[imgid_jpg]

	for imgid, idx in val_imgid2idx.items():
		imgid_jpg = imgid
		vgg_val_imgid2idx[imgid] = vgg_imgid2idx[imgid_jpg]

	pf = open("data/flickr30k/maf_train_imgid2idx.pkl", "wb")
	pickle.dump(vgg_train_imgid2idx, pf)
	pf.close()

	pf = open("data/flickr30k/maf_test_imgid2idx.pkl", "wb")
	pickle.dump(vgg_test_imgid2idx, pf)
	pf.close()

	pf = open("data/flickr30k/maf_val_imgid2idx.pkl", "wb")
	pickle.dump(vgg_val_imgid2idx, pf)
	pf.close()

	print(len(vgg_train_imgid2idx), len(vgg_test_imgid2idx), len(vgg_val_imgid2idx))

	print("done")


if __name__ == "__main__":
	split()