package classification;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ResampleSimpleClassification extends BasicClassification{

	public ResampleSimpleClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult1[] = new double[9];
		double validationResult2[] = new double[9];
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(classifier, randomSeed, "over");
			updateResult(validationResult1, eval);
		}
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(classifier, randomSeed, "under");
			updateResult(validationResult2, eval);
		}
		return getResult("oversample", classifier_name, validationResult1, times) + getResult(",undersample", classifier_name, validationResult2, times);
	}
}