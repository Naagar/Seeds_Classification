# confusion_matrix
import torch


class confusion_matrix():
	def __init__(self, model, num_classes, validation_loader):

		Print('Confusion Matrix')
		confusion_matrix = torch.zeros(num_classes, num_classes)
		with torch.no_grad():
		    for i, (inputs, classes) in enumerate(validation_loader):
		        inputs = inputs.to(device)
		        classes = classes.to(device)
		        outputs = model(inputs)
		        _, preds = torch.max(outputs, 1)
		        for t, p in zip(classes.view(-1), preds.view(-1)):
		                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)



# @torch.no_grad()
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([])
#     for batch in loader:
#         images, labels = batch

#         preds = model(images)
#         all_preds = torch.cat(
#             (all_preds, preds)
#             ,dim=0
#         )
#     return all_preds
# with torch.no_grad():
#     prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=256)
#     train_preds = get_all_preds(network, prediction_loader)


# preds_correct = get_num_correct(train_preds, train_set.targets)

# print('total correct:', preds_correct)
# print('accuracy:', preds_correct / len(train_set) * 100)

# train_set.targets
# train_preds.argmax(dim=1)

# stacked = torch.stack(
#     (
#         train_set.targets
#         ,train_preds.argmax(dim=1)
#     )
#     ,dim=1
# )

# stacked.shape

# stacked

# stacked[0].tolist()

# cmt = torch.zeros(4,4, dtype=torch.int64)

# for p in stacked:
#     tl, pl = p.tolist()
#     cmt[tl, pl] = cmt[tl, pl] + 1
# cmt

# import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix
# from resources.plotcm import plot_confusion_matrix

# cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
# print(type(cm))
# cm

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# from plotcm import plot_confusion_matrix

# plt.figure(figsize=(4, 4))
# plot_confusion_matrix(cm, train_set.classes)
