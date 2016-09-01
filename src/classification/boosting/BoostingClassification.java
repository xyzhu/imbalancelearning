package classification.boosting;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class BoostingClassification extends BasicClassification{

	public BoostingClassification(Instances data) {
		super(data);
	}


	//using bagging classification method
	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		AdaBoostM1 boost_classifier = new AdaBoostM1(); //set the classifier as bagging
		boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(boost_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("boost", classifier_name, validationResult, times);
	}

}
