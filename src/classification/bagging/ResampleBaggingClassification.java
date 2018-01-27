package classification.bagging;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;

public class ResampleBaggingClassification extends BasicClassification{

	public ResampleBaggingClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult1[] = new double[9];
		double validationResult2[] = new double[9];
		Bagging bag_classifier = new Bagging();
		bag_classifier.setClassifier(classifier);
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "over");
			updateResult(validationResult1, eval);
		}
		
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "under");
			updateResult(validationResult2, eval);
		}
		return getResult("overbag", classifier_name, validationResult1, times) + getResult(",underbag", classifier_name, validationResult2, times);
	}
}
