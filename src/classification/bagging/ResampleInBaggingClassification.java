package classification.bagging;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import Classifier.OverBagging;
import Classifier.SmoteBagging;
import Classifier.UnderBagging;
import Classifier.UnderOverBagging;

public class ResampleInBaggingClassification extends BasicClassification{

	public ResampleInBaggingClassification(Instances data) {
		super(data);
	}

	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception{
		String predictResult = "";
		predictResult += getOverBagClassificationResult(classifier, classifier_name, times);
		predictResult += getUnderBagClassificationResult(classifier, classifier_name, times);
		predictResult += getUnderOverBagClassificationResult(classifier, classifier_name, times);
		//predictResult += getSmoteBagClassificationResult(classifier, classifier_name, times);
		return predictResult;
	}

	public String getSmoteBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		SmoteBagging bag_classifier = new SmoteBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",smotebag", classifier_name, validationResult, times);

	}

	//using bagging classification method with under sampling
	public String getUnderBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		UnderBagging bag_classifier = new UnderBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",underbag", classifier_name, validationResult, times);

	}


	private String getOverBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		OverBagging bag_classifier = new OverBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("overbag", classifier_name, validationResult, times);

	}


	private String getUnderOverBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		UnderOverBagging bag_classifier = new UnderOverBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",underoverbag", classifier_name, validationResult, times);

	}

}
