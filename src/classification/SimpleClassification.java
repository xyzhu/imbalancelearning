package classification;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class SimpleClassification extends BasicClassification{

	public SimpleClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("simple", classifier_name, validationResult, times);
	}
}
