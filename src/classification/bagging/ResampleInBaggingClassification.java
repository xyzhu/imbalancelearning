package classification.bagging;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import Classifier.OverInBagging;
import Classifier.SmoteInBagging;
import Classifier.UnderInBagging;
import Classifier.UnderOverInBagging;
//import Classifier.UnderOverBaggingOld;

public class ResampleInBaggingClassification extends BasicClassification{

	public ResampleInBaggingClassification(Instances data) {
		super(data);
	}

	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception{
		String predictResult = "";
		predictResult += getOverBagClassificationResult(classifier, classifier_name, times);
		predictResult += getUnderBagClassificationResult(classifier, classifier_name, times);
		//predictResult += getUnderOverBagClassificationResult(classifier, classifier_name, times);
		//predictResult += getSmoteBagClassificationResult(classifier, classifier_name, times);
		return predictResult;
	}

	public String getSmoteBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		SmoteInBagging bag_classifier = new SmoteInBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",smoteinbag", classifier_name, validationResult, times);

	}

	//using bagging classification method with under sampling
	public String getUnderBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		UnderInBagging bag_classifier = new UnderInBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",underinbag", classifier_name, validationResult, times);

	}


	private String getOverBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		OverInBagging bag_classifier = new OverInBagging(); //set the classifier as bagging
		bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("overinbag", classifier_name, validationResult, times);

	}


	private String getUnderOverBagClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		UnderOverInBagging bag_classifier = new UnderOverInBagging(); //set the classifier as bagging
		bag_classifier.setBaseClassifier(classifier);//set the base classifier of bagging
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(bag_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("underoverinbag", classifier_name, validationResult, times);

	}

}
