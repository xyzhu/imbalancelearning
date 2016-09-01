package classification.bagging;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;

public class BaggingClassification extends BasicClassification{

	public BaggingClassification(Instances data) {
		super(data);
	}


	//using bagging classification method
	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Bagging bag_classifier = new Bagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging	
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("bag", classifier_name, validationResult, times);
	}

}
