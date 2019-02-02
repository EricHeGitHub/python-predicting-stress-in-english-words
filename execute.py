from numpy import mean
from sklearn.metrics import f1_score
import helper
import submission

training_data = helper.read_data('./asset/training_data.txt')
classifier_path = './asset/classifier.dat'
submission.train(training_data, classifier_path)
test_data = helper.read_data('./asset/tiny_test.txt')
prediction = submission.test(test_data, classifier_path)  
print(prediction) 
ground_truth = [1, 1, 2, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
print(f1_score(ground_truth, prediction, average='macro'))          

#training_data = helper.read_data('./asset/training_data.txt')
#test_data = helper.read_data('./asset/tiny_test.txt')
#classifier_path = './asset/classifier.dat'
#train(training_data, classifier_path)
#prediction = test(test_data, classifier_path)
#print(prediction)
#ground_truth = [1, 1, 2, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
#print(f1_score(ground_truth, prediction, average='macro'))